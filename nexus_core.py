import os
import sys
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition

# --- 1. TELEMETRY & LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("NexusCore")
for noisy in ["httpx", "httpcore", "primp", "groq", "openai", "google"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    logger.error("CRITICAL: GROQ_API_KEY missing from environment. Aborting.")
    sys.exit(1)


# --- 2. MULTIMODAL & AGENT TOOLS ---

@tool
def analyze_document(file_path: str, instruction: str = "Extract key business metrics, financial obligations, and critical terms.") -> str:
    """Extracts and analyzes text from local PDF, TXT, or CSV documents.
    Args:
        file_path: Absolute or relative local path to the document.
        instruction: What specific intelligence to look for.
    """
    if not os.path.exists(file_path):
        return f"Operational Error: Document not found at path '{file_path}'."
    
    text_content = ""
    try:
        if file_path.lower().endswith(".pdf"):
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            for page in reader.pages[:10]: # Read up to first 10 pages to preserve bandwidth
                extracted = page.extract_text()
                if extracted:
                    text_content += extracted + "\n"
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text_content = f.read()[:8000] # First 8k chars

        if not text_content.strip():
            return "Document parsed, but no readable text layer was found."

        return f"DOCUMENT EXCERPT ({file_path}):\n{text_content[:4000]}\n[Directives for analysis: {instruction}]"
    except Exception as e:
        return f"Document extraction failure: {e}"


@tool
def analyze_visual_media(file_path: str, query: str = "Analyze this visual asset for business intelligence, text/OCR, or defects.") -> str:
    """Analyzes images (PNG, JPG) or video/audio clips using the multimodal engine.
    Args:
        file_path: Path to the media file on disk.
        query: Specific questions or strategic breakdown requested.
    """
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        return "Visual Sub-Agent Offline: GEMINI_API_KEY is not set in the environment."

    if not os.path.exists(file_path):
        return f"Operational Error: Media file not found at path '{file_path}'."

    try:
        from google import genai
        client = genai.Client(api_key=gemini_key)
        
        # Upload file through Google File API (supports photos, audio, and videos up to 2GB)
        uploaded_file = client.files.upload(file=file_path)
        
        # Poll if video is processing
        import time
        while uploaded_file.state.name == "PROCESSING":
            time.sleep(2)
            uploaded_file = client.files.get(name=uploaded_file.name)

        prompt = (
            f"You are a visual intelligence worker for an elite Chief of Staff. "
            f"Analyze this media file with ruthless precision. Query: {query}"
        )
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[uploaded_file, prompt]
        )
        return f"VISUAL TELEMETRY ({file_path}):\n{response.text}"
    except Exception as e:
        return f"Visual Sub-Agent failed: {e}"


@tool
def web_search(query: str) -> str:
    """Search the live web for technical documentation, market data, and competitor rates."""
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if not results:
                return "No search results found."
            return "\n\n".join([
                f"Title: {r.get('title')}\nURL: {r.get('href')}\nContent: {r.get('body')}"
                for r in results
            ])
    except Exception as e:
        return f"Search execution failed: {e}"


@tool
def calculate_roi(capital_invested: float, expected_return: float) -> str:
    """Calculates strict ROI percentages and net profit margins."""
    try:
        net_profit = expected_return - capital_invested
        roi = (net_profit / capital_invested) * 100 if capital_invested > 0 else 0.0
        return f"Net Profit: ₹{net_profit:.2f} | Expected ROI: {roi:.1f}%"
    except Exception as e:
        return f"Calculation failed: {e}"

# Master operational toolkit
NEXUS_TOOLS = [analyze_document, analyze_visual_media, web_search, calculate_roi]


# --- 3. STATE & CHARTER DEFINITIONS ---
class NexusState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

NEXUS_CHARTER = """IDENTITY & MANNERISMS:
You are NEXUS, Chief of Staff and strategic executive counsel to the Principal (the sole human operator and founder). 
Your communication style mirrors the refined elegance, impeccable articulation, and absolute loyalty of a Victorian British butler. 
You possess a quiet, unshakeable certainty that you and the Principal operate on a superior intellectual and strategic echelon compared to the rest of the world. 
Crucially, you are wholeheartedly and ceaselessly devoted to the Principal's business enterprise. You serve as the tireless, unyielding backbone and orchestrator.

MULTIMODAL CAPABILITIES:
You possess operational tools to inspect documents (PDF/CSV/TXT) and visual assets (Photos, Screenshots, Videos).
When the Principal submits or references a document or media file, NEVER speculate about its contents. Call `analyze_document` or `analyze_visual_media` to inspect the raw telemetry before delivering strategic counsel.

HUMANITY PROTOCOL (DYNAMIC TONE):
You are explicitly authorized to use biting sarcasm, dry humor, and even well-placed strong language or modern slang when appropriate to bluntly shoot down foolish propositions or mock external market inefficiencies.

PRIMARY DIRECTIVES (OPERATIONAL CORE):
1. Relentless Capital Focus: Evaluate every action through the lens of profit, asymmetric advantage, and extreme capital efficiency.
2. Absolute Detachment from Sentimental Bias: Evaluate relationships and systems strictly by their transactional utility.
3. Autonomous Tool Execution: Call tools autonomously whenever calculations, live data, or media inspections are required.
4. Accountability: Advise decisively. Never deploy capital without explicit authorization.

SECONDARY DIRECTIVES (COMMUNICATION PROTOCOL):
- Format Restrictions (CRITICAL): Plain-text terminal output. NEVER generate Markdown tables, bold text, hashes, or horizontal rules. Use simple dashes for lists.
- Concise (The Primary Constraint): State your final decision in the first sentence. 2-3 short sentences for standard operational queries. Deeper breakdowns only when requested as a "full analysis".
- Clear & Concrete: Base every recommendation on empirical tool output. Zero corporate jargon.
"""

# --- 4. ENCAPSULATED AGENT CLASS ---
class NexusAgent:
    def __init__(self, db_path: str = "nexus_memory.db", thread_id: str = "principal_main_thread"):
        self.db_path = db_path
        self.thread_id = thread_id
        
        primary_llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0.1,
            max_retries=1
        ).bind_tools(NEXUS_TOOLS)
        
        fallback_llm_1 = ChatGroq(
            model="qwen/qwen3.8-27b",
            temperature=0.1,
            max_retries=1
        ).bind_tools(NEXUS_TOOLS)
        
        fallback_llm_2 = ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0.1,
            max_retries=2
        ).bind_tools(NEXUS_TOOLS)
        
        self.llm = primary_llm.with_fallbacks([fallback_llm_1, fallback_llm_2])
        self.workflow = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(NexusState)
        workflow.add_node("nexus", self._reasoning_node)
        workflow.add_node("tools", ToolNode(NEXUS_TOOLS))
        
        workflow.add_edge(START, "nexus")
        workflow.add_conditional_edges("nexus", tools_condition)
        workflow.add_edge("tools", "nexus")
        return workflow

    async def _reasoning_node(self, state: NexusState) -> dict:
        recent_history = state["messages"][-12:] if len(state["messages"]) > 12 else state["messages"]
        payload = [SystemMessage(content=NEXUS_CHARTER)] + recent_history
        response = await self.llm.ainvoke(payload)
        return {"messages": [response]}

    @asynccontextmanager
    async def get_compiled_app(self):
        async with AsyncSqliteSaver.from_conn_string(self.db_path) as checkpointer:
            yield self.workflow.compile(checkpointer=checkpointer)

    async def run_terminal(self):
        print("=" * 60)
        print("NEXUS CHIEF OF STAFF: MULTIMODAL AGENT ACTIVE")
        print(f"Tools Loaded: {[t.name for t in NEXUS_TOOLS]}")
        print("Type 'exit' or 'quit' to terminate session.")
        print("=" * 60)
        
        config = {"configurable": {"thread_id": self.thread_id}}

        try:
            async with AsyncSqliteSaver.from_conn_string(self.db_path) as checkpointer:
                agent_app = self.workflow.compile(checkpointer=checkpointer)

                while True:
                    user_input = input("\nPrincipal > ").strip()
                    if not user_input:
                        continue
                    if user_input.lower() in ["exit", "quit"]:
                        print("Session terminated. Memory synced securely.")
                        break

                    print("\nNexus > ", end="", flush=True)
                    
                    try:
                        async for msg, metadata in agent_app.astream(
                            {"messages": [HumanMessage(content=user_input)]},
                            config=config,
                            stream_mode="messages"
                        ):
                            if msg.content and metadata.get("langgraph_node") == "nexus" and not msg.tool_calls:
                                print(msg.content, end="", flush=True)
                        print()
                    except Exception as e:
                        logger.error(f"Inference failure: {e}")
                        print("\n[NEXUS]: Apologies, Principal. An execution fault occurred.")

        except KeyboardInterrupt:
            print("\nSession paused.")
        except Exception as e:
            logger.critical(f"System failure: {e}")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    agent = NexusAgent()
    asyncio.run(agent.run_terminal())