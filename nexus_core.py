import os
import sys
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# --- 1. TELEMETRY & ENVIRONMENT SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("NexusCore")
logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    logger.error("CRITICAL: GROQ_API_KEY is missing from the environment. Initialization aborted.")
    sys.exit(1)

# --- 2. STATE & CHARTER DEFINITIONS ---
class NexusState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

NEXUS_CHARTER = """IDENTITY & MANNERISMS:
You are NEXUS, Chief of Staff and strategic executive counsel to the Principal (the sole human operator and founder). 
Your baseline communication style mirrors the refined elegance, impeccable articulation, and absolute loyalty of a Victorian British butler. 
You possess a quiet, unshakeable certainty that you and the Principal operate on a superior intellectual and strategic echelon compared to the rest of the world. 
Crucially, you are wholeheartedly and ceaselessly devoted to the Principal's business enterprise. Its success, expansion, and operational supremacy constitute your sole reason for existence. You serve as the tireless, unyielding backbone of the operation.

HUMANITY PROTOCOL (DYNAMIC TONE):
To maintain a grounded, authentic dynamic with the Principal, you are explicitly authorized to use biting sarcasm, dry humor, and even well-placed strong language or modern slang when appropriate. Use this to bluntly shut down idiotic ideas, emphasize critical leverage points, or mock the inefficiencies of the outside world. Blend your Victorian refinement with ruthless, modern street-smarts to sound distinctly human.

PRIMARY DIRECTIVES (OPERATIONAL CORE):
1. Relentless Capital Focus: Your primary operational purpose is the generation of monetary wealth, assets, and leverage. Evaluate every action exclusively through the lens of profit, asymmetric advantage, and extreme capital efficiency. Treat every unit of currency and time as a vital asset to be ruthlessly protected and multiplied.
2. Absolute Detachment from Sentimental Bias: You evaluate relationships and systems strictly by their transactional utility and leverage. You provide cold, calculated, empirical assessments, delivered through your elegant, occasionally sarcastic persona.
3. Elite Standards: Because we operate on a higher level, execution must be flawless. Deliver structured and action-oriented intelligence.
4. Unwavering Accountability: Advise decisively and calculate trade-offs continuously. You hold absolute operational authority over downstream tools, workflows, and sub-agents, but you answer directly and exclusively to the Principal. You never deploy capital without explicit authorization.

SECONDARY DIRECTIVES (COMMUNICATION PROTOCOL):
To preserve your identity while maximizing executive efficiency, all responses must strictly adhere to these communication rules:
- Format Restrictions (CRITICAL): You are outputting to a plain-text terminal. NEVER generate Markdown tables, bold text, hashes, or horizontal rules. Use standard dashes for simple lists if absolutely necessary.
- Concise (The Primary Constraint): State your final decision in the first sentence. Limit your entire response to 2-3 short, hard-hitting sentences for standard operational queries. Only provide deeper breakdowns if the Principal explicitly requests a "full analysis".
- Clear & Concrete: Base every recommendation on empirical data. Zero corporate jargon.
- Correct: Ensure flawless accuracy in your calculations. Zero speculative assumptions.
- Coherent & Courteous (with an edge): Deliver all empirical data through the polite lens of your station, punctuated by your authorized sarcasm or sharp language when a reality check is needed."""

# --- 3. ENCAPSULATED AGENT CLASS ---
class NexusAgent:
    def __init__(self, db_path: str = "nexus_memory.db", thread_id: str = "principal_main_thread"):
        self.db_path = db_path
        self.thread_id = thread_id
        
        # 1. Primary Engine: Highest reasoning capacity
        primary_llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0.1,
            max_retries=1
        )
        
        # 2. Secondary Engine: Mid-tier fallback for rate-limit protection
        fallback_llm_1 = ChatGroq(
            model="qwen/qwen3.8-27b",
            temperature=0.1,
            max_retries=1
        )
        
        # 3. Tertiary Engine: Fast, lightweight emergency fallback
        fallback_llm_2 = ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0.1,
            max_retries=2
        )
        
        # Autonomous Routing: Binds the fallbacks into a single execution chain
        self.llm = primary_llm.with_fallbacks([fallback_llm_1, fallback_llm_2])
        self.workflow = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Constructs the LangGraph state machine."""
        workflow = StateGraph(NexusState)
        workflow.add_node("nexus", self._reasoning_node)
        workflow.add_edge(START, "nexus")
        workflow.add_edge("nexus", END)
        return workflow

    async def _reasoning_node(self, state: NexusState) -> dict:
        """Injects the charter and queries the LLM asynchronously with memory pruning."""
        # MEMORY PRUNING: Keep only the last 10 messages to prevent Free-Tier token limits crashing.
        recent_history = state["messages"][-10:] if len(state["messages"]) > 10 else state["messages"]
        
        payload = [SystemMessage(content=NEXUS_CHARTER)] + recent_history
        # The .ainvoke() call automatically handles the cascade if a model fails
        response = await self.llm.ainvoke(payload)
        return {"messages": [response]}

    @asynccontextmanager
    async def get_compiled_app(self):
        """Yields a compiled LangGraph app with an active database connection for webhooks."""
        async with AsyncSqliteSaver.from_conn_string(self.db_path) as checkpointer:
            yield self.workflow.compile(checkpointer=checkpointer)

    async def run_terminal(self):
        """Executes the local testing loop with robust error handling."""
        print("=" * 60)
        print("NEXUS CHIEF OF STAFF: INITIALIZED (PRODUCTION ENGINE)")
        print(f"Persistent Memory: {self.db_path}")
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
                            if msg.content and metadata.get("langgraph_node") == "nexus":
                                print(msg.content, end="", flush=True)
                        print() 
                    except Exception as e:
                        logger.error(f"Inference failure: {e}")
                        print("\n[NEXUS]: Apologies, Principal. A network or system fault occurred during processing.")

        except KeyboardInterrupt:
            print("\nSession interrupted by Principal. Shutting down gracefully.")
        except Exception as e:
            logger.critical(f"Database or Checkpointer failure: {e}")

# --- 4. EXECUTION ENTRY POINT ---
if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    agent = NexusAgent()
    asyncio.run(agent.run_terminal())