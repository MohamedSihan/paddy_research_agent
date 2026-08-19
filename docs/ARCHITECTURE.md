# Architecture and Agent Design

## Agent 1 — ResearchPlanner
**Pattern:** Planner–Executor.

Input: research question.

Output: `AgentMessage(message_type="RESEARCH_PLAN")` containing research goal, subquestions, retrieval queries and metrics.

## Agent 2 — EvidenceAnalyst
**Pattern:** Tool-using RAG.

The analyst receives the planner message, calls the local `EvidenceRetriever`, builds an evidence context, and produces a draft. It is instructed to cite source IDs and not fabricate experimental results.

## Agent 3 — QualityReviewer
**Pattern:** Reflection / critic.

The reviewer receives the draft and evidence IDs and checks citation coverage, qualification of comparisons, unsupported claims and limitations. It returns a structured review.

## Communication protocol

```text
RESEARCH_REQUEST
   ↓
ResearchPlanner
   ↓ AgentMessage(RESEARCH_PLAN)
EvidenceRetriever
   ↓ EvidenceBundle
EvidenceAnalyst
   ↓ AgentMessage(DRAFT_REPORT)
QualityReviewer
   ↓ AgentMessage(REVIEWED_REPORT)
Streamlit UI
```

The structured message schema is implemented with Pydantic and includes sender, recipient, message type, payload and trace ID.
