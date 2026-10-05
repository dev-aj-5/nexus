import os
import uuid
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
for noisy in ["httpx", "httpcore", "primp", "groq", "openai", "google"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_BASE_URL")
PRINCIPAL_TG_ID = os.getenv("PRINCIPAL_TG_ID")

DOWNLOAD_DIR = "/tmp/nexus_downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# --- 2. NEXUS ENGINE & BOT ---
agent = NexusAgent(db_path="nexus_memory.db", thread_id="principal_telegram_thread")
tg_app = Application.builder().token(TELEGRAM_TOKEN).build() if TELEGRAM_TOKEN else None


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if PRINCIPAL_TG_ID and str(update.effective_user.id) != PRINCIPAL_TG_ID:
        return
    await update.message.reply_text("Nexus Chief of Staff online. At your service, Principal.")


async def _execute_agent_pipeline(user_prompt: str) -> str:
    """Helper to stream response from LangGraph."""
    async with agent.get_compiled_app() as compiled_agent:
        config = {"configurable": {"thread_id": agent.thread_id}}
        response_text = ""
        try:
            async for msg, metadata in compiled_agent.astream(
                {"messages": [HumanMessage(content=user_prompt)]},
                config=config,
                stream_mode="messages"
            ):
                if msg.content and metadata.get("langgraph_node") == "nexus" and not msg.tool_calls:
                    response_text += msg.content

            if not response_text:
                response_text = "Operational directives executed, Principal."
        except Exception as e:
            logger.error(f"Inference error: {e}")
            response_text = "Apologies, Principal. A network or execution fault occurred."

        return response_text


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if PRINCIPAL_TG_ID and str(update.effective_user.id) != PRINCIPAL_TG_ID:
        return
    user_text = update.message.text
    if not user_text:
        return

    reply = await _execute_agent_pipeline(user_text)
    await update.message.reply_text(reply)


async def handle_multimedia(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Downloads attached documents, images, or videos and directs Nexus to inspect them."""
    if PRINCIPAL_TG_ID and str(update.effective_user.id) != PRINCIPAL_TG_ID:
        return

    message = update.message
    caption = message.caption or "Analyze this file thoroughly for key strategic data."
    
    file_obj = None
    file_ext = ""

    if message.document:
        file_obj = await context.bot.get_file(message.document.file_id)
        file_ext = os.path.splitext(message.document.file_name)[1] if message.document.file_name else ".pdf"
    elif message.photo:
        # Highest resolution photo is last item in list
        file_obj = await context.bot.get_file(message.photo[-1].file_id)
        file_ext = ".jpg"
    elif message.video:
        file_obj = await context.bot.get_file(message.video.file_id)
        file_ext = ".mp4"

    if not file_obj:
        return

    # Download to disk
    local_filename = f"{uuid.uuid4().hex[:8]}{file_ext}"
    local_path = os.path.join(DOWNLOAD_DIR, local_filename)
    await file_obj.download_to_drive(custom_path=local_path)

    # Formulate directive for Nexus
    directive = (
        f"The Principal has provided a file saved at '{local_path}'. "
        f"Instruction/Caption: '{caption}'. "
        f"Use your tools (analyze_document or analyze_visual_media) to inspect it and report your strategic findings."
    )

    reply = await _execute_agent_pipeline(directive)
    await message.reply_text(reply)

    # Clean up local file after inspection
    try:
        if os.path.exists(local_path):
            os.remove(local_path)
    except Exception:
        pass


if tg_app:
    tg_app.add_handler(CommandHandler("start", start_command))
    tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    # Capture photos, documents, and videos
    tg_app.add_handler(MessageHandler(filters.PHOTO | filters.Document.ALL | filters.VIDEO, handle_multimedia))


# --- 4. FASTAPI LIFESPAN & ROUTES ---
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
    return {"status": "operational", "system": "NEXUS_CHIEF_OF_STAFF"}

@app.post("/telegram-webhook")
async def telegram_webhook(request: Request):
    if not tg_app:
        return Response(status_code=status.HTTP_400_BAD_REQUEST)
    req_data = await request.json()
    update = Update.de_json(data=req_data, bot=tg_app.bot)
    await tg_app.process_update(update)
    return Response(status_code=status.HTTP_200_OK)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="warning")