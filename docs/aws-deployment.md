# AWS Deployment Architecture

This document describes how the Clinical AI Agent would be deployed to production on AWS. The prototype runs locally; this is the production migration path for a clinical-grade, HIPAA-compliant system.

---

## Architecture Diagram

```
                           ┌──────────────────┐
                           │   Route 53        │
                           │   (DNS failover)  │
                           └────────┬─────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
              ┌─────▼──────┐                 ┌──────▼─────┐
              │  us-east-1  │                 │  us-west-2  │
              │  (primary)  │                 │  (DR/standby)│
              └─────┬──────┘                 └──────┬──────┘
                    │                               │
                    ▼                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PRIMARY REGION (us-east-1)                         │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  VPC (HIPAA-compliant)                                       │    │
│  │  ├── Private Subnets (3 AZs: us-east-1a, 1b, 1c)           │    │
│  │  ├── No public subnets for compute                           │    │
│  │  ├── NAT Gateway (outbound only, for Bedrock API)           │    │
│  │  └── VPC Endpoints (S3, CloudWatch, ECR, Bedrock, STS)      │    │
│  │                                                              │    │
│  │  ┌───────────────────┐    ┌─────────────────────────────┐   │    │
│  │  │  ALB (internal)   │    │  API Gateway (external)      │   │    │
│  │  │  ├── TLS 1.2+     │    │  ├── WAF (OWASP rules)      │   │    │
│  │  │  ├── Health checks │    │  ├── Rate limiting           │   │    │
│  │  │  └── Path routing  │    │  ├── API key auth            │   │    │
│  │  └────────┬──────────┘    │  └── Throttling               │   │    │
│  │           │               └──────────────────────────────┘   │    │
│  │           │                                                  │    │
│  │  ┌────────▼──────────┐    ┌─────────────────────────────┐   │    │
│  │  │  ECS Fargate       │    │  ECS Fargate                │   │    │
│  │  │  Agent Service     │    │  Model Service              │   │    │
│  │  │  ├── LangChain     │───▶│  ├── FastAPI + XGBoost      │   │    │
│  │  │  ├── 2-10 tasks    │    │  ├── /predict /health /ready│   │    │
│  │  │  ├── 3 AZ spread   │    │  ├── 2-10 tasks, 3 AZ      │   │    │
│  │  │  └── Secrets Mgr   │    │  └── ECR image, no secrets  │   │    │
│  │  └────────┬──────────┘    └─────────────────────────────┘   │    │
│  │           │                                                  │    │
│  │  ┌────────▼──────────┐    ┌─────────────────────────────┐   │    │
│  │  │  Amazon Bedrock    │    │  OpenSearch Serverless       │   │    │
│  │  │  ├── Claude Sonnet │    │  ├── Vector search (RAG)    │   │    │
│  │  │  ├── Titan Embed v2│    │  ├── Encryption at rest     │   │    │
│  │  │  ├── Prompt caching│    │  └── VPC endpoint access    │   │    │
│  │  │  └── Provisioned   │    └─────────────────────────────┘   │    │
│  │  │      throughput    │                                      │    │
│  │  └───────────────────┘                                       │    │
│  │                                                              │    │
│  │  ┌───────────────────┐    ┌─────────────────────────────┐   │    │
│  │  │  S3 (versioned)    │    │  CloudWatch                 │   │    │
│  │  │  ├── Model artifacts│    │  ├── Inference logs         │   │    │
│  │  │  ├── Training data  │    │  ├── SLO dashboards         │   │    │
│  │  │  ├── Clinical docs  │    │  ├── Drift alerts           │   │    │
│  │  │  ├── Audit trail    │    │  ├── Anomaly detection      │   │    │
│  │  │  │   (Object Lock)  │    │  └── PagerDuty integration  │   │    │
│  │  │  └── Cross-region   │    └─────────────────────────────┘   │    │
│  │  │      replication    │                                      │    │
│  │  └───────────────────┘                                       │    │
│  │                                                              │    │
│  │  ┌───────────────────┐    ┌─────────────────────────────┐   │    │
│  │  │  ECR               │    │  Step Functions              │   │    │
│  │  │  ├── Model image   │    │  ├── ML training pipeline    │   │    │
│  │  │  ├── Agent image   │    │  ├── Data validation         │   │    │
│  │  │  └── Cross-region  │    │  ├── Model evaluation        │   │    │
│  │  │      replication   │    │  ├── Champion/challenger      │   │    │
│  │  └───────────────────┘    │  └── Automated retraining     │   │    │
│  │                           └─────────────────────────────┘   │    │
│  │  ┌───────────────────┐    ┌─────────────────────────────┐   │    │
│  │  │  Secrets Manager   │    │  AWS KMS                     │   │    │
│  │  │  ├── DB credentials│    │  ├── Customer-managed CMKs   │   │    │
│  │  │  ├── API keys      │    │  ├── Auto-rotation           │   │    │
│  │  │  └── Auto-rotation │    │  └── Per-service key policy  │   │    │
│  │  └───────────────────┘    └─────────────────────────────┘   │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │  Security & Compliance Layer                                   │   │
│  │  ├── CloudTrail (all API calls, multi-region)                 │   │
│  │  ├── AWS Config (compliance rules, drift detection)           │   │
│  │  ├── GuardDuty (threat detection)                             │   │
│  │  ├── Security Hub (centralized findings)                      │   │
│  │  ├── Inspector (container vulnerability scanning)             │   │
│  │  └── Macie (PHI discovery in S3)                              │   │
│  └───────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Component Migration: Local → AWS

| Local Component | AWS Service | Why |
|---|---|---|
| ChromaDB (in-process) | Amazon OpenSearch Serverless | Managed vector search, auto-scaling, encrypted, backed up |
| FastAPI in Docker (Colima) | ECS Fargate | Serverless containers, auto-scaling, no server management |
| HuggingFace embeddings (local) | Bedrock Titan Embed v2 | Managed, scalable, consistent with Bedrock ecosystem |
| Bedrock Claude (API call) | Same (Bedrock Claude) | No change — already using Bedrock |
| File-based model registry | S3 + DynamoDB | Versioned artifacts in S3, metadata in DynamoDB |
| JSONL inference logs | CloudWatch Logs | Managed, queryable, alarms, dashboards |
| PSI drift detection (local) | CloudWatch Metrics + Lambda | Scheduled Lambda computes PSI, publishes to CloudWatch |
| Performance benchmarks | CodePipeline + CloudWatch | Run in CI, publish metrics to CloudWatch |
| Clinical documents (local files) | S3 (versioned, cross-region replicated) | Source of truth for RAG documents |
| Environment variables | Secrets Manager | Auto-rotation, encrypted, audit logged |

---

## Security Architecture

### Defense in Depth (7 Layers)

```
Layer 1: Network
  ├── WAF on API Gateway (OWASP Top 10, SQL injection, XSS)
  ├── DDoS protection (Shield Standard, automatic)
  ├── Rate limiting (API Gateway throttling)
  └── IP allowlisting for clinician access

Layer 2: Network Isolation
  ├── Private subnets only (no public-facing compute)
  ├── VPC endpoints for all AWS services (no internet traversal)
  ├── Security groups: least-privilege, deny-by-default
  └── Network ACLs as backup to security groups

Layer 3: Authentication & Authorization
  ├── API Gateway: API key + OAuth2 for external access
  ├── IAM Roles: ECS task roles (no long-lived credentials)
  ├── AWS SSO: human access through identity provider
  └── Bedrock: scoped IAM policies per model

Layer 4: Encryption
  ├── At rest: KMS customer-managed keys (CMK) for all services
  ├── In transit: TLS 1.2+ mandatory, no exceptions
  ├── S3: SSE-KMS with bucket policy enforcing encryption
  └── ECS: encrypted ephemeral storage

Layer 5: Application Security
  ├── Input validation (Pydantic + clinical range checks)
  ├── Circuit breaker (prevents cascading failures)
  ├── Prompt injection protection (system prompt constraints)
  ├── No patient identifiers in logs (anonymized features only)
  └── Pre-commit hooks: bandit security scan on every commit

Layer 6: Audit & Compliance
  ├── CloudTrail: every API call logged, immutable
  ├── Inference audit trail: S3 Object Lock (WORM)
  ├── AWS Config: compliance rules enforced automatically
  └── FDA 21 CFR Part 11: electronic signatures, access controls

Layer 7: Threat Detection
  ├── GuardDuty: anomalous API calls, compromised credentials
  ├── Inspector: container image vulnerability scanning
  ├── Macie: PHI discovery and classification in S3
  └── Security Hub: centralized security findings dashboard
```

### Secrets Management

| Secret | Storage | Rotation |
|---|---|---|
| Database credentials | Secrets Manager | Auto-rotate every 30 days |
| API keys (external) | Secrets Manager | Auto-rotate every 90 days |
| KMS keys | AWS KMS | Auto-rotate annually |
| ECS task credentials | IAM task roles | Temporary, auto-refreshed |
| Bedrock access | IAM role (no keys) | N/A — role-based |

### Container Security

- **ECR image scanning:** Automatic vulnerability scanning on push
- **Inspector:** Continuous scanning of running containers
- **Base image:** `python:3.11-slim` — minimal attack surface
- **No root:** Container runs as non-root user
- **Read-only filesystem:** Mount only required paths as writable
- **Resource limits:** CPU/memory caps prevent resource exhaustion

---

## HIPAA Compliance

### Technical Safeguards

| Requirement | Implementation |
|---|---|
| Access control | IAM roles, API Gateway auth, no shared credentials |
| Audit controls | CloudTrail, inference logging, S3 Object Lock |
| Integrity controls | KMS encryption, S3 versioning, immutable audit trail |
| Transmission security | TLS 1.2+, VPC endpoints, no public internet |
| Encryption at rest | KMS CMK for all data stores |

### Administrative Safeguards
- **BAA (Business Associate Agreement)** with AWS — required before storing PHI
- **All services must be HIPAA-eligible** (Bedrock, ECS, S3, OpenSearch, CloudWatch — all are)
- **Access reviews:** Quarterly review of IAM policies and access logs
- **Incident response plan:** Documented in runbook with escalation path

### FDA 21 CFR Part 11 Awareness
- **Audit trails:** Every prediction immutably logged with timestamp, user, input, output
- **Electronic signatures:** Clinician must acknowledge AI recommendations
- **Access controls:** Role-based, logged, time-limited
- **System validation:** CI/CD pipeline validates model quality before deployment

---

## Disaster Recovery

### RPO/RTO Targets

| Scenario | RPO (data loss) | RTO (recovery time) |
|---|---|---|
| Single AZ failure | 0 (multi-AZ) | 0 (automatic failover) |
| Service degradation | 0 | < 5 min (circuit breaker + graceful degradation) |
| Region failure | < 1 hour | < 4 hours (DR region activation) |
| Data corruption | 0 (versioned, Object Lock) | < 1 hour (restore from version) |
| Model corruption | 0 (registry rollback) | < 5 min (promote previous champion) |

### Multi-AZ (Default — Automatic)

```
us-east-1a          us-east-1b          us-east-1c
┌──────────┐        ┌──────────┐        ┌──────────┐
│ ECS Task │        │ ECS Task │        │ ECS Task │
│ (model)  │        │ (model)  │        │ (agent)  │
└──────────┘        └──────────┘        └──────────┘
     │                   │                   │
     └───────────────────┴───────────────────┘
                         │
                    ALB (cross-AZ)
```

- ECS tasks spread across 3 AZs automatically
- ALB health checks route traffic away from unhealthy AZs
- OpenSearch Serverless is multi-AZ by default
- S3 is multi-AZ by default (99.999999999% durability)
- **Single AZ failure = zero downtime, zero data loss**

### Multi-Region DR (us-east-1 → us-west-2)

```
PRIMARY (us-east-1)                    DR (us-west-2)
┌─────────────────────┐                ┌─────────────────────┐
│ ECS (active)        │                │ ECS (standby, 0 tasks│
│ OpenSearch (active) │                │    — scale to 2 on   │
│ S3 (source)    ────────────────────▶ │    activation)        │
│ ECR (source)   ────────────────────▶ │ S3 (replica)          │
│ Bedrock (active)    │                │ ECR (replica)         │
│ CloudWatch (active) │                │ Bedrock (available)   │
└─────────────────────┘                └─────────────────────┘

Route 53 DNS failover:
  primary health check fails → switch DNS to DR region
  DR region scales up ECS tasks → re-index OpenSearch from S3
  RTO: ~4 hours (OpenSearch re-indexing is the bottleneck)
```

**Cross-Region Replication:**

| Resource | Replication Method | Lag |
|---|---|---|
| S3 (model artifacts, clinical docs, audit logs) | S3 Cross-Region Replication | < 15 min |
| ECR (container images) | ECR replication rules | < 5 min |
| Secrets Manager | Multi-region secrets | Real-time |
| DynamoDB (model registry) | Global Tables | Real-time |
| OpenSearch | Not replicated — rebuild from S3 | ~2-3 hours on activation |

**DR Activation Runbook:**
1. Route 53 health check detects primary region failure
2. DNS automatically fails over to DR region
3. EventBridge triggers DR activation Lambda
4. Lambda scales ECS tasks from 0 → 2 in DR region
5. Lambda triggers OpenSearch re-indexing from S3 documents
6. Lambda verifies /health and /ready endpoints
7. Notify operations team via PagerDuty
8. Monitor DR region for stability for 1 hour
9. Begin root cause analysis on primary region

### Backup Strategy

| Data | Backup Method | Retention | Recovery |
|---|---|---|---|
| Model artifacts (.pkl, .pt) | S3 versioning | 90 days | Restore specific version |
| Training data | S3 versioning | 1 year | Restore and retrain |
| Clinical documents | S3 versioning + cross-region | 7 years (regulatory) | Restore + re-ingest |
| Inference audit trail | S3 Object Lock (WORM) | 7 years (FDA) | Immutable, always available |
| Model registry (metadata) | DynamoDB with PITR | 35 days continuous | Point-in-time restore |
| Vector store | Rebuildable from S3 documents | N/A | Re-index from source |
| CloudWatch logs | Log retention policy | 1 year | Query directly |

---

## ML Training Pipeline (Step Functions)

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│ S3 Event     │───▶│ Data         │───▶│ Train        │
│ (new data    │    │ Validation   │    │ (XGBoost +   │
│  uploaded)   │    │ ├── Schema   │    │  PyTorch)    │
│              │    │ ├── Ranges   │    │              │
│              │    │ └── Quality  │    │              │
└──────────────┘    └──────────────┘    └──────┬───────┘
                                               │
┌──────────────┐    ┌──────────────┐    ┌──────▼───────┐
│ Monitor      │◀───│ Deploy       │◀───│ Evaluate     │
│ (24h watch,  │    │ (promote or  │    │ ├── AUC gate │
│  auto-revert │    │  rollback)   │    │ ├── Recall   │
│  if degrade) │    │              │    │ ├── Latency  │
└──────────────┘    └──────────────┘    │ └── Shadow   │
                                        │     test     │
                                        └──────────────┘
```

---

## Auto-Scaling Strategy

### Model Service (ECS Fargate)
- **Min:** 2 tasks (high availability across AZs)
- **Max:** 10 tasks
- **Scale-out:** CPU > 70% for 3 minutes OR request count > 100/min
- **Scale-in:** CPU < 30% for 10 minutes
- **Scale-in cooldown:** 5 minutes (prevent flapping)
- **Scale-in protection:** Never scale below 2 during business hours

### Agent Service (ECS Fargate)
- **Min:** 2 tasks
- **Max:** 5 tasks
- **Scale trigger:** Concurrent request count > 50

### OpenSearch Serverless
- Scales automatically based on query load — no configuration needed

### Bedrock
- **On-demand:** Default, scales with request volume
- **Provisioned throughput:** For predictable workloads (e.g., 50 req/min guaranteed)
- Use provisioned for CardioMEMS daily batch processing (predictable volume)

---

## Cost Estimation (Monthly)

### Pilot (< 1,000 patients)

| Service | Configuration | Estimated Cost |
|---|---|---|
| ECS Fargate (model) | 2 tasks × 0.5 vCPU × 1GB | ~$30 |
| ECS Fargate (agent) | 2 tasks × 0.5 vCPU × 1GB | ~$30 |
| Bedrock Claude Sonnet | ~100K requests/month | ~$150 |
| Bedrock Titan Embed | ~500K embeddings/month | ~$10 |
| OpenSearch Serverless | 2 OCU minimum | ~$350 |
| S3 + CloudWatch + KMS | Storage + logs + keys | ~$50 |
| NAT Gateway | Outbound traffic | ~$35 |
| Secrets Manager | 10 secrets | ~$5 |
| **Pilot Total** | | **~$660/month** |

### Production (100,000+ patients)

| Service | Configuration | Estimated Cost |
|---|---|---|
| ECS Fargate (model) | 4-10 tasks, auto-scaling | ~$200 |
| ECS Fargate (agent) | 2-5 tasks | ~$100 |
| Bedrock Claude (provisioned) | ~1M requests/month | ~$1,500 |
| Bedrock Titan Embed | ~5M embeddings/month | ~$100 |
| OpenSearch Serverless | Auto-scaled | ~$700 |
| S3 + CloudWatch + KMS | At scale | ~$200 |
| DR Region (standby) | Minimal (S3 replication + 0 ECS) | ~$100 |
| WAF + Shield | Standard protection | ~$50 |
| **Production Total** | | **~$2,950/month** |

**Cost optimization strategies:**
- Use Savings Plans for ECS Fargate (up to 50% savings for 1-year commit)
- Bedrock prompt caching reduces input token costs by ~93%
- Route simple queries to Haiku (10x cheaper than Sonnet)
- pgvector in Aurora instead of OpenSearch Serverless saves ~$300/month at low scale

---

## Deployment Strategy

### Blue/Green for Model Service
1. Build new container image with updated model → push to ECR
2. Deploy new ECS task definition as "green" service
3. Run health checks + smoke tests against green
4. Switch ALB target group from blue → green
5. Keep blue running for 30 minutes (instant rollback)
6. Deregister blue if green is healthy

### Canary for Agent Service
1. Deploy new agent version to 10% of traffic (ALB weighted target groups)
2. Monitor error rate, response quality, and latency for 15 minutes
3. If metrics are healthy → roll out to 50% → 100%
4. If degradation detected → rollback to previous version instantly

### Model Version Promotion
1. New model trained → registered as "candidate" in registry
2. Run quality gates (AUC > 0.75, recall > 0.60, latency < 100ms)
3. Shadow test: run candidate alongside champion for 24 hours, compare predictions
4. If candidate wins → promote via registry API
5. Monitor for 24 hours → if degradation → automatic rollback via CloudWatch alarm

### Rollback Procedures
| Scenario | Action | Time |
|---|---|---|
| Bad model deployment | Registry rollback (swap champion/challenger) | < 1 min |
| Bad agent deployment | ALB target group switch back to blue | < 1 min |
| Bad RAG update | S3 version restore + re-index OpenSearch | < 30 min |
| Bad prompt change | Git revert + redeploy agent service | < 10 min |
| Region failure | Route 53 DNS failover to DR | < 5 min (automatic) |

---

## Monitoring & Alerting

### CloudWatch Dashboards

**Model Service Dashboard:**
- Prediction latency (p50, p95, p99)
- Prediction volume (requests/min)
- Error rate
- CPU/memory utilization
- Circuit breaker state

**ML Health Dashboard:**
- AUC trend (daily evaluation)
- Feature drift PSI scores
- Prediction distribution (mean, std)
- Model version in production
- Retraining pipeline status

**Business Dashboard:**
- Patients assessed per day
- High-risk patients flagged
- Clinician review rate
- Agent response time

### Alert Escalation

| Alert | Threshold | Action | Notify |
|---|---|---|---|
| p99 latency > 200ms | 5 min sustained | Auto-scale model service | On-call engineer |
| Error rate > 0.1% | 3 min sustained | Page on-call | Engineering lead |
| AUC < 0.75 | Daily evaluation | Trigger retraining pipeline | ML team |
| PSI > 0.2 (any feature) | Hourly check | Alert + investigation | ML team |
| Circuit breaker OPEN | Immediate | Activate graceful degradation | On-call + clinical team |
| Prediction confidence < 50% (trend) | 100 consecutive | Investigate model health | ML team |
| Region health check fail | 3 consecutive | Route 53 DNS failover | Operations team |
