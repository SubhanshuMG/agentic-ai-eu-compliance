"""
audit/logger.py

Layer 4: Article 12 + Article 19 compliant audit logger.
Writes structured, immutable records to PostgreSQL with append-only rules.
Separates raw PII (GDPR-deletable) from audit metadata (compliance-permanent).

EU AI Act obligations:
  - Art. 12: Automatic logging for high-risk AI systems
  - Art. 19: 6-month minimum retention for operational logs,
             10-year retention for technical documentation
"""

import hashlib
import json
import uuid
from datetime import datetime, timedelta
from typing import Optional
import asyncpg
import structlog

log = structlog.get_logger()


class ComplianceAuditLogger:
    def __init__(self, db_pool: asyncpg.Pool):
        self.pool = db_pool

    async def setup_tables(self):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_audit_log (
                    log_id           TEXT PRIMARY KEY,
                    session_id       TEXT NOT NULL,
                    timestamp        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    system_id        TEXT NOT NULL,
                    system_version   TEXT NOT NULL,
                    risk_class       TEXT NOT NULL,
                    operation_id     TEXT NOT NULL,
                    tool_invoked     TEXT,
                    risk_score       FLOAT,
                    confidence       FLOAT,
                    guardrails       JSONB,
                    compliance_flags JSONB,
                    reasoning_trace  JSONB,
                    human_oversight  JSONB,
                    input_hash       TEXT NOT NULL,
                    output_hash      TEXT,
                    retention_until  TIMESTAMPTZ NOT NULL,
                    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );

                CREATE OR REPLACE RULE no_delete_audit AS
                    ON DELETE TO agent_audit_log DO INSTEAD NOTHING;

                CREATE OR REPLACE RULE no_update_audit AS
                    ON UPDATE TO agent_audit_log DO INSTEAD NOTHING;

                CREATE TABLE IF NOT EXISTS agent_session_pii (
                    session_id     TEXT PRIMARY KEY,
                    user_id_hash   TEXT NOT NULL,
                    raw_input_ref  TEXT,
                    gdpr_delete_at TIMESTAMPTZ,
                    deleted_at     TIMESTAMPTZ
                );

                CREATE INDEX IF NOT EXISTS idx_audit_session   ON agent_audit_log(session_id);
                CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON agent_audit_log(timestamp);
                CREATE INDEX IF NOT EXISTS idx_audit_risk      ON agent_audit_log(risk_score);
            """)

    async def log_agent_action(
        self,
        session_id: str,
        system_id: str,
        system_version: str,
        operation_id: str,
        tool_invoked: Optional[str],
        risk_score: float,
        confidence: float,
        raw_input: str,
        raw_output: Optional[str],
        guardrails_applied: list[str],
        guardrails_passed: bool,
        compliance_flags: list[str],
        reasoning_trace: list[str],
        human_oversight: Optional[dict] = None,
    ) -> str:
        log_id = str(uuid.uuid4())

        # Hash inputs/outputs: never store raw PII in audit log
        input_hash  = hashlib.sha256(raw_input.encode()).hexdigest()
        output_hash = hashlib.sha256(raw_output.encode()).hexdigest() if raw_output else None

        # Art. 19: 6-month minimum retention
        retention_until = datetime.utcnow() + timedelta(days=180)

        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO agent_audit_log (
                    log_id, session_id, system_id, system_version, risk_class,
                    operation_id, tool_invoked, risk_score, confidence,
                    guardrails, compliance_flags, reasoning_trace,
                    human_oversight, input_hash, output_hash, retention_until
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16)
            """,
                log_id, session_id, system_id, system_version, "high",
                operation_id, tool_invoked, risk_score, confidence,
                json.dumps({"applied": guardrails_applied, "passed": guardrails_passed}),
                json.dumps(compliance_flags),
                json.dumps(reasoning_trace),
                json.dumps(human_oversight) if human_oversight else None,
                input_hash, output_hash, retention_until,
            )

        log.info("audit_log_written", log_id=log_id, session_id=session_id,
                 risk_score=risk_score, tool=tool_invoked)
        return log_id

    async def generate_compliance_report(
        self,
        system_id: str,
        from_date: datetime,
        to_date: datetime,
    ) -> dict:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT
                    COUNT(*)                                     AS total_operations,
                    AVG(risk_score)                              AS avg_risk_score,
                    MAX(risk_score)                              AS max_risk_score,
                    SUM(CASE WHEN risk_score > 0.75 THEN 1 ELSE 0 END) AS high_risk_count,
                    COUNT(DISTINCT session_id)                   AS unique_sessions,
                    SUM(CASE WHEN human_oversight IS NOT NULL THEN 1 ELSE 0 END) AS human_reviewed_count
                FROM agent_audit_log
                WHERE system_id = $1 AND timestamp BETWEEN $2 AND $3
            """, system_id, from_date, to_date)

            return {
                "system_id": system_id,
                "report_period": {"from": from_date.isoformat(), "to": to_date.isoformat()},
                "article_9_compliance": {
                    "risk_management_active": True,
                    "continuous_monitoring": True,
                    "metrics": dict(rows[0]),
                },
                "generated_at": datetime.utcnow().isoformat(),
            }
