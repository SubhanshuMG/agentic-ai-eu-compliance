# agentic-ai-eu-compliance

> A production-grade, six-layer compliance architecture for agentic AI systems under EU AI Act Article 9.

Companion code for the article **"Governing the Ungovernable: How to Build a Compliance Framework for Agentic AI That Satisfies EU AI Act Article 9 (And Actually Works)"** by Subhanshu Mohan Gupta.

---

## Repo Structure

```
agentic-ai-eu-compliance/
├── orchestration/
│   └── agent_graph.py          # LangGraph stateful agent with checkpointing + runtime risk scoring
├── guardrails/
│   └── middleware.py           # Prompt injection, PII redaction, output groundedness guards
├── audit/
│   └── logger.py               # Article 12/19-compliant append-only audit logger (PostgreSQL)
├── observability/
│   └── telemetry.py            # OpenTelemetry GenAI semantic conventions instrumentation
├── cicd/
│   └── compliance_gate.py      # Article 9 CI/CD gate with Giskard + adversarial test suite
├── api/
│   └── main.py                 # FastAPI integration layer wiring all six layers together
├── tests/
│   └── compliance/
│       └── test_article9.py    # Full Article 9 compliance test suite
├── .github/
│   └── workflows/
│       └── ai-compliance.yml   # GitHub Actions CI/CD pipeline with compliance gate
├── .env.example                # Environment variable template
├── requirements.txt            # All dependencies
└── README.md
```

---

## Article 9 Obligation Coverage

| Article 9 Requirement | File | Layer |
|----------------------|------|-------|
| Continuous iterative process (Art. 9(2)) | `orchestration/agent_graph.py` | L1 |
| Identify foreseeable risks (Art. 9(2a)) | `guardrails/middleware.py` | L2 |
| Estimate risk under misuse (Art. 9(2b)) | `orchestration/agent_graph.py` | L3 |
| Post-market monitoring (Art. 9(2c)) | `observability/telemetry.py` | L6 |
| Risk mitigation by design (Art. 9(5)) | `guardrails/middleware.py` | L2 |
| Testing with defined metrics (Art. 9(6)) | `cicd/compliance_gate.py` | CI/CD |
| Human oversight (Art. 14) | `orchestration/agent_graph.py` | L5 |
| Automatic logging (Art. 12) | `audit/logger.py` | L4 |
| 6-month retention (Art. 19) | `audit/logger.py` | L4 |

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/agentic-ai-eu-compliance.git
cd agentic-ai-eu-compliance

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Fill in your API keys

# 4. Start PostgreSQL (Docker)
docker run -d --name compliance-db \
  -e POSTGRES_PASSWORD=testpass \
  -e POSTGRES_DB=compliance_db \
  -p 5432:5432 postgres:16

# 5. Run the API
uvicorn api.main:app --reload

# 6. Run compliance tests
pytest tests/compliance/ -v
```

---

## Stack

| Layer | Tool |
|-------|------|
| Orchestration | LangGraph, LangSmith |
| Guardrails | NeMo Guardrails, Lakera Guard, Guardrails AI |
| Risk Scoring | Custom runtime scorer |
| Audit Logging | PostgreSQL (append-only), structlog |
| Human-in-the-Loop | LangGraph interrupt middleware |
| Observability | OpenTelemetry GenAI, Prometheus, Grafana |
| Testing | Giskard, pytest-asyncio, custom adversarial suite |
| CI/CD | GitHub Actions |
| Compliance Mgmt | Credo AI, MLflow |

---

## License

MIT. See [LICENSE](LICENSE).

---

*EU AI Act Article 9 high-risk obligations apply from August 2, 2026 for Annex III systems.*
