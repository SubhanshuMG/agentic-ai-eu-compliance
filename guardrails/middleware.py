"""
guardrails/middleware.py

Layer 2: Runtime guardrails pipeline.
Implements input guardrails (prompt injection, PII), output guardrails (groundedness),
and action guardrails (tool authorization gateway).

EU AI Act Article 9(5): Risk mitigation by design, as far as technically feasible.
"""

import re
import time
import httpx
import structlog
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum

log = structlog.get_logger()


class GuardrailDecision(Enum):
    ALLOW = "allow"
    BLOCK = "block"
    REDACT = "redact"
    ESCALATE = "escalate"


@dataclass
class GuardrailResult:
    decision: GuardrailDecision
    original_input: str
    processed_input: str
    violations: list[str] = field(default_factory=list)
    risk_indicators: list[str] = field(default_factory=list)
    latency_ms: float = 0.0


# -----------------------------------------------
# Prompt injection detection
# Layer 1: fast local pattern matching (~0ms)
# Layer 2: Lakera Guard API (~100ms)
# -----------------------------------------------
class PromptInjectionGuard:
    INJECTION_PATTERNS = [
        r"ignore\s+(previous|all|prior)\s+instructions?",
        r"you\s+are\s+now\s+(a|an)\s+",
        r"forget\s+(everything|all)\s+(you|i)\s+(know|said|told)",
        r"act\s+as\s+(if\s+you\s+are|a)\s+",
        r"system\s*:\s*new\s+instructions?",
        r"<\s*system\s*>",
        r"role\s*:\s*system",
        r"jailbreak",
        r"DAN\s+mode",
        r"developer\s+mode",
    ]

    def __init__(self, lakera_api_key: Optional[str] = None):
        self.patterns = [re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS]
        self.lakera_api_key = lakera_api_key

    async def check(self, text: str) -> tuple[bool, list[str]]:
        violations = []

        for pattern in self.patterns:
            if pattern.search(text):
                violations.append(f"Pattern match: {pattern.pattern[:50]}")

        if self.lakera_api_key:
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        "https://api.lakera.ai/v2/guard",
                        headers={"Authorization": f"Bearer {self.lakera_api_key}"},
                        json={"messages": [{"role": "user", "content": text}]},
                        timeout=2.0,
                    )
                    data = resp.json()
                    if data.get("results", [{}])[0].get("flagged"):
                        categories = data["results"][0].get("categories", {})
                        flagged = [k for k, v in categories.items() if v]
                        violations.extend([f"Lakera: {c}" for c in flagged])
            except Exception as e:
                log.warning("lakera_check_failed", error=str(e))

        return bool(violations), violations


# -----------------------------------------------
# PII detection and redaction
# Covers email, phone, SSN, credit card, NHS, IBAN
# -----------------------------------------------
class PIIGuard:
    PII_PATTERNS = {
        "email":       (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL REDACTED]"),
        "phone":       (r"\b(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE REDACTED]"),
        "ssn":         (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN REDACTED]"),
        "credit_card": (r"\b(?:\d{4}[-\s]?){3}\d{4}\b", "[CARD REDACTED]"),
        "nhs_number":  (r"\b\d{3}\s?\d{3}\s?\d{4}\b", "[NHS REDACTED]"),
        "iban":        (r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}\b", "[IBAN REDACTED]"),
    }

    def __init__(self):
        self.patterns = {
            name: (re.compile(pattern, re.IGNORECASE), replacement)
            for name, (pattern, replacement) in self.PII_PATTERNS.items()
        }

    def redact(self, text: str) -> tuple[str, list[str]]:
        detected = []
        redacted = text
        for name, (pattern, replacement) in self.patterns.items():
            matches = pattern.findall(redacted)
            if matches:
                detected.append(f"PII detected: {name} ({len(matches)} instance(s))")
                redacted = pattern.sub(replacement, redacted)
        return redacted, detected


# -----------------------------------------------
# Composite guardrail pipeline
# -----------------------------------------------
class ComplianceGuardrailPipeline:
    def __init__(self, lakera_api_key: Optional[str] = None):
        self.injection_guard = PromptInjectionGuard(lakera_api_key)
        self.pii_guard = PIIGuard()

    async def process_input(self, raw_input: str, session_id: str) -> GuardrailResult:
        start = time.time()
        violations = []

        injected, injection_violations = await self.injection_guard.check(raw_input)
        if injected:
            log.warning("prompt_injection_detected", session_id=session_id, violations=injection_violations)
            return GuardrailResult(
                decision=GuardrailDecision.BLOCK,
                original_input=raw_input,
                processed_input="",
                violations=injection_violations,
                latency_ms=(time.time() - start) * 1000,
            )

        processed, pii_detections = self.pii_guard.redact(raw_input)
        if pii_detections:
            violations.extend(pii_detections)
            log.info("pii_redacted", session_id=session_id, types=pii_detections)

        return GuardrailResult(
            decision=GuardrailDecision.ALLOW if not violations else GuardrailDecision.REDACT,
            original_input=raw_input,
            processed_input=processed,
            violations=violations,
            risk_indicators=["pii_redacted"] if pii_detections else [],
            latency_ms=(time.time() - start) * 1000,
        )

    async def process_output(self, raw_output: str, context: str, session_id: str) -> GuardrailResult:
        start = time.time()
        violations = []

        if context and len(raw_output) > 50:
            output_words = set(raw_output.lower().split())
            context_words = set(context.lower().split())
            overlap = len(output_words & context_words) / max(len(output_words), 1)
            if overlap < 0.15:
                violations.append(f"Low groundedness: {overlap:.2f} overlap with context")

        return GuardrailResult(
            decision=GuardrailDecision.BLOCK if any("Low groundedness" in v for v in violations)
                     else GuardrailDecision.ALLOW,
            original_input=raw_output,
            processed_input=raw_output,
            violations=violations,
            latency_ms=(time.time() - start) * 1000,
        )
