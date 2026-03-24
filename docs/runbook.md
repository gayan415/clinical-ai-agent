# Operational Runbook — Clinical AI Agent

## Model Service Latency Spike

**Symptoms:** p99 latency exceeds 200ms SLO, predictions slow down

**Diagnosis:**
1. Check inference monitor: `python -c "from mlops.monitor import InferenceMonitor; m = InferenceMonitor('logs'); print(m.get_metrics_summary())"`
2. Check container resources: `docker stats clinical-ai-model`
3. Check for input data anomalies (unusually large payloads?)

**Remediation:**
1. If container resource-constrained → scale up (more CPU/memory)
2. If input data anomaly → check upstream data pipeline
3. If model itself is slow → rollback to previous model version via registry
4. If persistent → retrain with optimized hyperparameters

---

## Prediction Drift Detected

**Symptoms:** PSI > 0.2 for one or more input features

**Diagnosis:**
1. Run drift check: `python -c "from mlops.drift import check_drift; ..."`
2. Identify which features drifted
3. Compare input distributions: training baseline vs current

**Remediation:**
1. Investigate cause: did the patient population change? New data source? Bug in data pipeline?
2. If population genuinely changed → retrain model on recent data
3. If data pipeline bug → fix pipeline, drift will resolve
4. Monitor for 24-48 hours after fix to confirm stability

---

## LLM Response Quality Degradation

**Symptoms:** Agent responses are incomplete, miss citations, or ignore safety rules

**Diagnosis:**
1. Check Bedrock service status (AWS Health Dashboard)
2. Review recent prompt changes
3. Check if system prompt was accidentally modified
4. Review tool call logs — is the agent calling tools correctly?

**Remediation:**
1. If Bedrock degraded → wait for AWS resolution, fall back to RAG-only mode
2. If prompt changed → rollback to previous prompt version (git revert)
3. If tool calls failing → check model service health, RAG index integrity

---

## RAG Returning Irrelevant Results

**Symptoms:** Queries return unrelated document chunks, retrieval quality tests failing

**Diagnosis:**
1. Run retrieval quality tests: `make test-ml`
2. Check if ChromaDB/vector store was corrupted
3. Verify embedding model is loaded correctly
4. Check if new documents were ingested with bad content

**Remediation:**
1. Re-index documents: `rm -rf chroma_db && make ingest`
2. If embedding model corrupted → re-download: `make download-model`
3. If bad document ingested → remove from rag/data/, re-ingest
4. Run retrieval quality tests to confirm fix

---

## Complete Service Outage

**Symptoms:** All requests failing, no predictions served

**Diagnosis:**
1. Check model container: `docker ps` — is it running?
2. Check health endpoint: `curl http://localhost:8000/health`
3. Check ready endpoint: `curl http://localhost:8000/ready`
4. Check circuit breaker state in agent logs

**Remediation:**
1. If container down → restart: `make docker-run`
2. If model not loaded → check MODEL_PATH, retrain if artifacts missing: `make train`
3. Activate graceful degradation — agent returns RAG-only results
4. Notify clinical team: "AI risk predictions temporarily unavailable, manual review required"
5. Page on-call engineer

---

## Escalation Path

1. **P3 (low):** Latency spike, minor drift → investigate during business hours
2. **P2 (medium):** Quality degradation, significant drift → investigate within 4 hours
3. **P1 (high):** Complete outage, agent providing incorrect recommendations → immediate response
4. **P0 (critical):** Patient safety risk — activate graceful degradation immediately, notify clinical leadership
