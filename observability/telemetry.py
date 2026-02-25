"""
observability/telemetry.py

Layer 6: OpenTelemetry instrumentation using GenAI semantic conventions (v1.37+).
Captures traces, metrics, and logs for Article 9(2c) post-market monitoring.

Metrics exposed:
  - gen_ai.agent.risk_score (histogram)
  - gen_ai.agent.guardrail_triggered (counter)
  - gen_ai.agent.human_review_required (counter)
  - gen_ai.agent.tool_latency (histogram)
"""

import time
import hashlib
from contextlib import contextmanager

from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.resources import Resource


def setup_telemetry(service_name: str, otlp_endpoint: str = "http://localhost:4317"):
    resource = Resource.create({
        "service.name":              service_name,
        "service.version":           "1.0.0",
        "deployment.environment":    "production",
        "ai.system.risk_class":      "high",
        "ai.system.regulation":      "EU_AI_ACT",
    })

    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint))
    )
    trace.set_tracer_provider(tracer_provider)

    metric_reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(endpoint=otlp_endpoint),
        export_interval_millis=30000,
    )
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)

    return trace.get_tracer(service_name), metrics.get_meter(service_name)


class AgentTelemetry:
    """
    Instruments agent operations using OpenTelemetry GenAI semantic conventions.
    Span attributes follow the gen_ai.* namespace from OTel spec v1.37+.
    """

    def __init__(self, service_name: str):
        self.tracer, self.meter = setup_telemetry(service_name)

        self.risk_score_histogram = self.meter.create_histogram(
            name="gen_ai.agent.risk_score",
            description="Risk score per agent action (Article 9 monitoring)",
            unit="1",
        )
        self.guardrail_counter = self.meter.create_counter(
            name="gen_ai.agent.guardrail_triggered",
            description="Count of guardrail trigger events",
            unit="1",
        )
        self.human_review_counter = self.meter.create_counter(
            name="gen_ai.agent.human_review_required",
            description="Count of actions requiring human oversight (Article 14)",
            unit="1",
        )
        self.tool_latency = self.meter.create_histogram(
            name="gen_ai.agent.tool_latency",
            description="Latency per tool invocation",
            unit="ms",
        )

    @contextmanager
    def trace_agent_turn(self, session_id: str, model: str, user_id: str):
        with self.tracer.start_as_current_span("gen_ai.agent.turn") as span:
            span.set_attribute("gen_ai.system",              "openai")
            span.set_attribute("gen_ai.request.model",       model)
            span.set_attribute("gen_ai.agent.session_id",    session_id)
            span.set_attribute("ai.compliance.user_id_hash",
                               hashlib.sha256(user_id.encode()).hexdigest()[:16])
            span.set_attribute("ai.compliance.regulation",   "EU_AI_ACT_2024_1689")
            span.set_attribute("ai.compliance.article",      "9")
            yield span

    @contextmanager
    def trace_tool_call(self, tool_name: str, risk_score: float):
        start = time.time()
        with self.tracer.start_as_current_span(f"gen_ai.agent.tool.{tool_name}") as span:
            span.set_attribute("gen_ai.tool.name",           tool_name)
            span.set_attribute("ai.risk.score",              risk_score)
            span.set_attribute("ai.risk.threshold_exceeded", risk_score > 0.75)
            self.risk_score_histogram.record(
                risk_score,
                {"tool": tool_name, "risk_tier": "high" if risk_score > 0.75 else "medium"}
            )
            yield span
            self.tool_latency.record(
                (time.time() - start) * 1000, {"tool": tool_name}
            )

    def record_guardrail_event(self, guardrail_type: str, decision: str, session_id: str):
        self.guardrail_counter.add(1, {
            "guardrail.type":     guardrail_type,
            "guardrail.decision": decision,
            "session_id":         session_id[:8],
        })

    def record_human_review(self, tool_name: str, risk_score: float):
        self.human_review_counter.add(1, {
            "tool":               tool_name,
            "risk_score_bucket":  f"{int(risk_score * 10) / 10:.1f}",
        })
