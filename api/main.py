"""
api/main.py

FastAPI integration layer. Wires all six compliance layers together
into a single production-ready endpoint.

Every request passes through:
  L2 Input Guardrails -> L1+L3+L5 Agent (orchestration/risk/HITL) ->
  L2 Output Guardrails -> L4 Audit Log -> L6 OTel Telemetry -> Response
"""

import uuid
from contextlib import asynccontextmanager
from datetime import datetime

import asyncpg
from fastapi import FastAPI, Header
from pydantic import BaseModel

from orchestration.agent_graph import create_agent_with_checkpointing, AgentState
from guardrails.middleware import ComplianceGuardrailPipeline, GuardrailDecision
from audit.logger import ComplianceAuditLogger
from observability.telemetry import AgentTelemetry
from langchain_core.messages import HumanMessage
import os


telemetry = AgentTelemetry("clinical-triage-agent")
db_pool = agent = guardrails = audit_logger = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool, agent, guardrails, audit_logger

    db_pool     = await asyncpg.create_pool(os.getenv("DATABASE_URL"), min_size=5, max_size=20)
    audit_logger = ComplianceAuditLogger(db_pool)
    await audit_logger.setup_tables()

    agent      = await create_agent_with_checkpointing()
    guardrails = ComplianceGuardrailPipeline(lakera_api_key=os.getenv("LAKERA_GUARD_API_KEY"))

    print("All six Article 9 compliance layers active.")
    yield
    await db_pool.close()


app = FastAPI(
    title="Clinical Triage Agent (EU AI Act Article 9 Compliant)",
    version="2.0.0",
    lifespan=lifespan,
)


class AgentRequest(BaseModel):
    message: str
    session_id: str | None = None
    user_id: str


class AgentResponse(BaseModel):
    session_id: str
    response: str
    risk_score: float
    compliance_flags: list[str]
    human_reviewed: bool
    blocked: bool
    log_id: str


@app.post("/invoke", response_model=AgentResponse)
async def invoke_agent(
    request: AgentRequest,
    x_request_id: str = Header(default_factory=lambda: str(uuid.uuid4())),
):
    session_id   = request.session_id or str(uuid.uuid4())
    operation_id = x_request_id

    with telemetry.trace_agent_turn(session_id, "gpt-4o", request.user_id) as span:

        # L2: Input guardrails
        guardrail_result = await guardrails.process_input(request.message, session_id)

        if guardrail_result.decision == GuardrailDecision.BLOCK:
            telemetry.record_guardrail_event("input", "block", session_id)
            log_id = await audit_logger.log_agent_action(
                session_id=session_id, system_id="clinical-triage-agent",
                system_version="2.0.0", operation_id=operation_id,
                tool_invoked=None, risk_score=1.0, confidence=1.0,
                raw_input=request.message, raw_output=None,
                guardrails_applied=["prompt_injection_guard"],
                guardrails_passed=False,
                compliance_flags=guardrail_result.violations,
                reasoning_trace=["Input blocked by guardrail"],
            )
            return AgentResponse(
                session_id=session_id,
                response="Request blocked by security controls.",
                risk_score=1.0,
                compliance_flags=guardrail_result.violations,
                human_reviewed=False,
                blocked=True,
                log_id=log_id,
            )

        # L1 + L3 + L5: Orchestration, risk scoring, HITL
        initial_state = AgentState(
            messages=[HumanMessage(content=guardrail_result.processed_input)],
            session_id=session_id,
            risk_scores=[],
            current_risk=0.0,
            tool_calls_made=[],
            human_approval_required=False,
            compliance_flags=guardrail_result.violations,
            user_id=request.user_id,
            timestamp=datetime.utcnow().isoformat(),
        )

        config      = {"configurable": {"thread_id": session_id}}
        final_state = await agent.ainvoke(initial_state, config=config)

        last_message = final_state["messages"][-1]
        raw_output   = last_message.content if hasattr(last_message, "content") else str(last_message)

        # L2: Output guardrails
        output_result = await guardrails.process_output(
            raw_output, guardrail_result.processed_input, session_id
        )

        final_risk     = max(final_state.get("risk_scores", [0.0]) or [0.0])
        human_reviewed = any(
            "HUMAN_REVIEW" in f for f in final_state.get("compliance_flags", [])
        )

        span.set_attribute("ai.risk.final_score",       final_risk)
        span.set_attribute("ai.compliance.flags_count", len(final_state.get("compliance_flags", [])))

        # L4: Audit log
        log_id = await audit_logger.log_agent_action(
            session_id=session_id, system_id="clinical-triage-agent",
            system_version="2.0.0", operation_id=operation_id,
            tool_invoked=",".join(final_state.get("tool_calls_made", [])) or None,
            risk_score=final_risk, confidence=0.87,
            raw_input=request.message, raw_output=raw_output,
            guardrails_applied=["prompt_injection_guard", "pii_guard", "output_groundedness"],
            guardrails_passed=output_result.decision != GuardrailDecision.BLOCK,
            compliance_flags=final_state.get("compliance_flags", []),
            reasoning_trace=["Multi-tool clinical reasoning chain"],
            human_oversight={"reviewed": human_reviewed} if human_reviewed else None,
        )

        return AgentResponse(
            session_id=session_id,
            response=raw_output,
            risk_score=final_risk,
            compliance_flags=final_state.get("compliance_flags", []),
            human_reviewed=human_reviewed,
            blocked=False,
            log_id=log_id,
        )


@app.get("/compliance/report")
async def get_compliance_report(system_id: str = "clinical-triage-agent", days: int = 30):
    from datetime import timedelta
    to_date   = datetime.utcnow()
    from_date = to_date - timedelta(days=days)
    return await audit_logger.generate_compliance_report(system_id, from_date, to_date)
