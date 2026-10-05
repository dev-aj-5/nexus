import os
import logging
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response, status
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from langchain_core.messages import HumanMessage
from nexus_core import NexusAgent

# --- 1. TELEMETRY & CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("NexusTelegram")
logging.getLogger("httpx").setLevel(logging.WARNING)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_BASE_URL")  # e.g., https://nexus-nzcw.onrender.com
PRINCIPAL_TG_ID = os.getenv("PRINCIPAL_TG_ID")  # Security lock

# --- 2. NEXUS & TELEGRAM INITIALIZATION ---
# Persistent memory database remains nexus_memory.db
agent = NexusAgent(db_path="nexus_memory.db", thread_id="principal_telegram_thread")
tg_app = Application.builder().token(TELEGRAM_TOKEN).build() if TELEGRAM_TOKEN else None

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if PRINCIPAL_TG_ID and str(update.effective_user.id) != PRINCIPAL_TG_ID:
        return  # Silently ignore unauthorized users
    await update.message.reply_text("Nexus Chief of Staff online. At your service, Principal.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Security lock: Verify sender is the Principal
    if PRINCIPAL_TG_ID and str(update.effective_user.id) != PRINCIPAL_TG_ID:
        return 
        
    user_text = update.message.text
    if not user_text:
        return

    # Safely open a database connection for this specific request
    async with agent.get_compiled_app() as compiled_agent:
        config = {"configurable": {"thread_id": agent.thread_id}}
        response_text = ""
        try:
            async for msg, metadata in compiled_agent.astream(
                {"messages": [HumanMessage(content=user_text)]},
                config=config,
                stream_mode="messages"
            ):
                if msg.content and metadata.get("langgraph_node") == "nexus":
                    response_text += msg.content

            if not response_text:
                response_text = "Operational directives processed, Principal."
        except Exception as e:
            logger.error(f"Inference error: {e}")
            response_text = "Apologies, Principal. A network or system fault occurred."

    await update.message.reply_text(response_text)

if tg_app:
    tg_app.add_handler(CommandHandler("start", start_command))
    tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

# --- 3. FASTAPI LIFESPAN & ROUTES ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    if tg_app:
        await tg_app.initialize()
        await tg_app.start()
        if WEBHOOK_URL:
            webhook_endpoint = f"{WEBHOOK_URL.rstrip('/')}/telegram-webhook"
            await tg_app.bot.set_webhook(url=webhook_endpoint)
            logger.info(f"Telegram webhook configured: {webhook_endpoint}")
    yield
    if tg_app:
        await tg_app.stop()
        await tg_app.shutdown()

app = FastAPI(title="Nexus Core Service", lifespan=lifespan)

@app.get("/")
@app.get("/health")
async def health_check():
    """Health check route targeted by ping services to keep the container warm."""
    return {"status": "operational", "system": "NEXUS_CHIEF_OF_STAFF"}

@app.post("/telegram-webhook")
async def telegram_webhook(request: Request):
    """Receives inbound messages from Telegram."""
    if not tg_app:
        return Response(status_code=status.HTTP_400_BAD_REQUEST)
    req_data = await request.json()
    update = Update.de_json(data=req_data, bot=tg_app.bot)
    await tg_app.process_update(update)
    return Response(status_code=status.HTTP_200_OK)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="warning")