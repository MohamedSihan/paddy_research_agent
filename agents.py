import os, json, re, requests
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from rag import EvidenceRetriever

class AgentMessage(BaseModel):
    sender: str
    recipient: str
    message_type: str
    payload: Dict[str, Any]
    trace_id: str

class LLMClient:
    def __init__(self):
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY")
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.or_model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        self.groq_model = os.getenv("GROQ_MODEL", "groq/compound")

    def chat(self, provider, system, user):
        if provider == "openrouter":
            key, base, model = self.openrouter_key, os.getenv("OPENROUTER_BASE_URL","https://openrouter.ai/api/v1"), self.or_model
        else:
            key, base, model = self.groq_key, os.getenv("GROQ_BASE_URL","https://api.groq.com/openai/v1"), self.groq_model
        
        if key:
            try:
                r = requests.post(
                    f"{base}/chat/completions",
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={"model": model, "messages":[{"role":"system","content":system},{"role":"user","content":user}],
                          "temperature":0.2},
                    timeout=90
                )
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            except Exception as err:
                print(f"[LLMClient] Request failed for {provider} ({model}): {err}")

        # Fallback provider if primary provider failed or has no key
        alt_provider = "openrouter" if provider != "openrouter" else "groq"
        if alt_provider == "openrouter":
            alt_key, alt_base, alt_model = self.openrouter_key, os.getenv("OPENROUTER_BASE_URL","https://openrouter.ai/api/v1"), self.or_model
        else:
            alt_key, alt_base, alt_model = self.groq_key, os.getenv("GROQ_BASE_URL","https://api.groq.com/openai/v1"), self.groq_model

        if alt_key:
            try:
                print(f"[LLMClient] Falling back to {alt_provider} ({alt_model})...")
                r = requests.post(
                    f"{alt_base}/chat/completions",
                    headers={"Authorization": f"Bearer {alt_key}", "Content-Type": "application/json"},
                    json={"model": alt_model, "messages":[{"role":"system","content":system},{"role":"user","content":user}],
                          "temperature":0.2},
                    timeout=90
                )
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            except Exception as err:
                print(f"[LLMClient] Fallback request failed for {alt_provider} ({alt_model}): {err}")

        return None

class ResearchPlanner:
    name = "ResearchPlanner"
    def run(self, question, trace_id, llm):
        system = """You are a research-planning agent for a university project on paddy disease detection,
lightweight CNNs, hybrid CNN-Vision Transformers, robustness, and edge deployment.
Return JSON with: research_goal, subquestions (3-5), retrieval_queries (3-5), required_metrics."""
        prompt = f"Research question: {question}"
        raw = llm.chat("groq", system, prompt) if llm else None
        if raw:
            try:
                data = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
            except Exception:
                data = self.fallback(question)
        else:
            data = self.fallback(question)
        return AgentMessage(sender=self.name, recipient="EvidenceAnalyst",
                            message_type="RESEARCH_PLAN", payload=data, trace_id=trace_id)

    def fallback(self, q):
        return {
            "research_goal": q,
            "subquestions": [
                "What evidence exists for lightweight CNNs in plant disease classification?",
                "What evidence exists for hybrid CNN–Transformer models?",
                "Which metrics matter for edge deployment and robustness?"
            ],
            "retrieval_queries": [q, "paddy disease dataset CNN", "hybrid CNN vision transformer edge deployment"],
            "required_metrics": ["accuracy", "macro-F1", "parameters", "FLOPs", "latency", "robustness"]
        }

class EvidenceAnalyst:
    name = "EvidenceAnalyst"
    def run(self, plan_msg, retriever, llm):
        evidence = retriever.multi_search(plan_msg.payload["retrieval_queries"], k_each=4)[:10]
        context = "\n\n".join(
            f"[{e['id']}] {e['title']} ({e['year']})\nURL: {e['url']}\n{e['text']}"
            for e in evidence
        )
        system = """You are an evidence-analysis agent. Use ONLY the supplied evidence cards for factual
claims. Produce a structured report with: executive_summary, findings, comparison_table, limitations,
and cited_sources. Cite sources using their IDs like [SRC-01]. Never invent experimental results."""
        prompt = f"Plan:\n{json.dumps(plan_msg.payload, indent=2)}\n\nEvidence:\n{context}"
        raw = llm.chat("openrouter", system, prompt) if llm else None
        report = raw if raw else self.fallback(evidence, plan_msg.payload["research_goal"])
        return AgentMessage(sender=self.name, recipient="QualityReviewer",
                            message_type="DRAFT_REPORT",
                            payload={"report": report, "evidence": evidence},
                            trace_id=plan_msg.trace_id)

    def fallback(self, evidence, goal):
        lines = [f"## Research support report\n\n**Question:** {goal}\n",
                 "### Evidence-grounded findings"]
        for e in evidence[:6]:
            lines.append(f"- **{e['title']}**: {e['text'].split('Summary:')[-1].strip()} [{e['id']}]")
        lines += ["\n### Limitations", "- Evidence cards summarize source metadata and should be checked against original papers.",
                  "- Cross-paper accuracy and latency are not directly comparable without identical protocols."]
        return "\n".join(lines)

class QualityReviewer:
    name = "QualityReviewer"
    def run(self, draft_msg, llm):
        report = draft_msg.payload["report"]
        evidence = draft_msg.payload["evidence"]
        ids = [e["id"] for e in evidence]
        system = """You are a strict academic reviewer. Check whether the draft uses only supplied sources,
whether claims have source IDs, whether comparisons are qualified, and whether limitations are acknowledged.
Return JSON with: approved, issues (array), improvements (array), score_0_to_10."""
        prompt = f"Available source IDs: {ids}\n\nDraft:\n{report}"
        raw = llm.chat("groq", system, prompt) if llm else None
        if raw:
            try:
                review = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
            except Exception:
                review = self.fallback(report, ids)
        else:
            review = self.fallback(report, ids)
        return AgentMessage(sender=self.name, recipient="StreamlitUI",
                            message_type="REVIEWED_REPORT",
                            payload={"report": report, "review": review, "evidence": evidence},
                            trace_id=draft_msg.trace_id)

    def fallback(self, report, ids):
        cited = sum(1 for x in ids if f"[{x}]" in report)
        return {"approved": cited >= min(3, len(ids)),
                "issues": [] if cited >= 3 else ["Increase source citations in the draft."],
                "improvements": ["Verify important claims against original sources.", "Report hardware/protocol details for latency comparisons."],
                "score_0_to_10": min(10, 5 + cited)}

def run_workflow(question):
    from uuid import uuid4
    trace = str(uuid4())
    retriever = EvidenceRetriever()
    llm = LLMClient()
    planner = ResearchPlanner()
    analyst = EvidenceAnalyst()
    reviewer = QualityReviewer()
    m1 = planner.run(question, trace, llm)
    m2 = analyst.run(m1, retriever, llm)
    m3 = reviewer.run(m2, llm)
    return m1, m2, m3
