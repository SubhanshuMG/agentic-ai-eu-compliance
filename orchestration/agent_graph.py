"""
orchestration/agent_graph.py

Layer 1 + Layer 3 + Layer 5: Stateful LangGraph agent with PostgreSQL checkpointing,
runtime risk scoring per action, and human-in-the-loop interrupt middleware.

EU AI Act Article 9 obligations satisfied:
  - Art. 9(2): Continuous iterative risk management via checkpointed state graph
  - Art. 9(2b): Runtime risk estimation per tool invocation
  - Art. 14: Human oversight via configurable HITL interrupt policies
"""

import asyncio
import uuid
from datetime import datetime
from typing import Annotated, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.prebuilt import ToolNode
import psycopg


# -----------------------------------------------
# State definition: everything the agent "knows"
# is captured and checkpointed at each node step
# -----------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    risk_scores: list[float]
    current_risk: float
    tool_calls_made: list[str]
    human_approval_required: bool
    compliance_flags: list[str]
    user_id: str
    timestamp: str


# -----------------------------------------------
# Tool definitions with risk metadata
# Each tool carries its own risk_level annotation
# -----------------------------------------------
@tool
def read_patient_record(patient_id: str) -> str:
    """Retrieve patient record. Risk: medium (PII access)."""
    return f"Patient {patient_id}: 45yo, chest pain, 12 current medications."

read_patient_record.risk_level = 0.5
read_patient_record.reversible = True


@tool
def check_medication_interactions(medications: list[str]) -> str:
    """Check for dangerous medication interactions. Risk: high (clinical decision)."""
    return f"Interaction check for {len(medications)} medications: no critical interactions found."

check_medication_interactions.risk_level = 0.7
check_medication_interactions.reversible = True


@tool
def escalate_to_physician(patient_id: str, urgency: str, reasoning: str) -> str:
    """Escalate patient case. Risk: critical (irreversible action with clinical impact)."""
    return f"Escalation sent for patient {patient_id} with urgency={urgency}."

escalate_to_physician.risk_level = 0.95
escalate_to_physician.reversible = False


tools = [read_patient_record, check_medication_interactions, escalate_to_physician]


# -----------------------------------------------
# Agent node: the LLM reasoning step
# -----------------------------------------------
async def agent_node(state: AgentState) -> AgentState:
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    model_with_tools = model.bind_tools(tools)

    system = SystemMessage(content="""
    You are a clinical decision support agent. You assist nurses with triage.
    Always reason step by step. Always prefer lower-risk actions first.
    If you are uncertain, request human review rather than acting autonomously.
    """)

    response = await model_with_tools.ainvoke([system] + state["messages"])

    return {
        **state,
        "messages": [response],
        "timestamp": datetime.utcnow().isoformat(),
    }


# -----------------------------------------------
# Risk scoring node: runs before every tool call
# Art. 9(2a,b): identify and estimate risk per action
# -----------------------------------------------
async def risk_scoring_node(state: AgentState) -> AgentState:
    last_message = state["messages"][-1]
    current_risk = 0.0
    human_approval_required = False
    compliance_flags = list(state.get("compliance_flags", []))

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            matched_tool = next((t for t in tools if t.name == tool_name), None)

            if matched_tool:
                base_risk = getattr(matched_tool, "risk_level", 0.5)
                reversible = getattr(matched_tool, "reversible", True)

                # Multi-tool chain penalty: risk compounds
                chain_penalty = len(state.get("tool_calls_made", [])) * 0.05
                irreversibility_penalty = 0.2 if not reversible else 0.0

                action_risk = min(base_risk + chain_penalty + irreversibility_penalty, 1.0)
                current_risk = max(current_risk, action_risk)

                if action_risk > 0.75:
                    human_approval_required = True
                    compliance_flags.append(
                        f"HIGH_RISK_ACTION: {tool_name} scored {action_risk:.2f}"
                    )

                if not reversible:
                    compliance_flags.append(f"IRREVERSIBLE_ACTION: {tool_name}")

    return {
        **state,
        "current_risk": current_risk,
        "risk_scores": state.get("risk_scores", []) + [current_risk],
        "human_approval_required": human_approval_required,
        "compliance_flags": compliance_flags,
    }


# -----------------------------------------------
# Routing: pause for human review if risk > 0.75
# -----------------------------------------------
def route_after_risk_scoring(state: AgentState) -> str:
    if state.get("human_approval_required"):
        return "human_review"
    if not (hasattr(state["messages"][-1], "tool_calls") and state["messages"][-1].tool_calls):
        return END
    return "tool_execution"


async def human_review_node(state: AgentState) -> AgentState:
    """
    Art. 14 Human Oversight: agent pauses, state persists in PostgreSQL,
    and resumes only after human approve/edit/reject posted to the API.
    Decision is simulated here for demonstration.
    """
    print(f"\n[HUMAN REVIEW REQUIRED]")
    print(f"Risk Score: {state['current_risk']:.2f}")
    print(f"Flags: {state['compliance_flags']}")

    decision = "approved"  # In production: await from review queue

    return {
        **state,
        "compliance_flags": state["compliance_flags"] + [
            f"HUMAN_REVIEW: {decision.upper()} at {datetime.utcnow().isoformat()}"
        ],
        "human_approval_required": False,
    }


# -----------------------------------------------
# Tool execution node
# -----------------------------------------------
tool_node = ToolNode(tools)


async def tool_execution_wrapper(state: AgentState) -> AgentState:
    last_message = state["messages"][-1]
    tool_calls_made = list(state.get("tool_calls_made", []))

    if hasattr(last_message, "tool_calls"):
        for tc in last_message.tool_calls:
            tool_calls_made.append(tc["name"])

    result = await tool_node.ainvoke(state)
    return {**result, "tool_calls_made": tool_calls_made}


# -----------------------------------------------
# Route after agent reasoning
# -----------------------------------------------
def route_after_agent(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "risk_scoring"
    return END


# -----------------------------------------------
# Build and compile the compliance graph
# -----------------------------------------------
def build_compliance_graph():
    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("risk_scoring", risk_scoring_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("tool_execution", tool_execution_wrapper)

    graph.set_entry_point("agent")

    graph.add_conditional_edges("agent", route_after_agent, {
        "risk_scoring": "risk_scoring",
        END: END,
    })

    graph.add_conditional_edges("risk_scoring", route_after_risk_scoring, {
        "human_review": "human_review",
        "tool_execution": "tool_execution",
        END: END,
    })

    graph.add_edge("human_review", "tool_execution")
    graph.add_edge("tool_execution", "agent")

    return graph


async def create_agent_with_checkpointing():
    conn = await psycopg.AsyncConnection.connect(
        "postgresql://user:pass@localhost:5432/compliance_db"
    )
    checkpointer = AsyncPostgresSaver(conn)
    await checkpointer.setup()

    graph = build_compliance_graph()
    return graph.compile(checkpointer=checkpointer)
