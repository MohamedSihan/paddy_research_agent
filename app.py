import os, json, uuid
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from agents import run_workflow
from rag import EvidenceRetriever

load_dotenv()
st.set_page_config(page_title="Paddy Research Agent", page_icon="🌾", layout="wide")

st.title("🌾 Paddy Disease AI Research Support Agent")
st.caption("Agentic RAG tool for lightweight CNN vs hybrid CNN–Vision Transformer research")

with st.sidebar:
    st.header("Project controls")
    st.write("**Domain:** Paddy disease detection + edge AI")
    st.write("**Corpus:** 24 curated evidence cards")
    st.write("**Patterns:** planner/executor · RAG tool-use · reflection")
    top_k = st.slider("Retrieved evidence shown", 3, 10, 6)

question = st.text_area(
    "Research question",
    "Compare lightweight CNNs and hybrid CNN–Vision Transformers for paddy disease detection on edge devices, focusing on accuracy, macro-F1, model size, FLOPs, robustness and latency.",
    height=110,
)

if st.button("Run agentic research", type="primary"):
    with st.spinner("Planning → retrieving → analysing → reviewing..."):
        m1, m2, m3 = run_workflow(question)
    st.session_state["result"] = (m1, m2, m3)

if "result" in st.session_state:
    m1, m2, m3 = st.session_state["result"]
    tabs = st.tabs(["Final report", "Agent trace", "Retrieved evidence", "Reviewer"])
    with tabs[0]:
        st.markdown(m3.payload["report"])
    with tabs[1]:
        st.json({"planner": m1.model_dump(), "analyst": m2.model_dump(), "reviewer": m3.model_dump()})
    with tabs[2]:
        for e in m3.payload["evidence"][:top_k]:
            with st.expander(f"{e['id']} · {e['title']} · score {e['score']:.3f}"):
                st.write(e["text"])
                st.markdown(f"Source: {e['url']}")
    with tabs[3]:
        st.json(m3.payload["review"])

st.divider()
st.subheader("Retrieval-only mode")
q = st.text_input("Test a corpus query", "hybrid CNN vision transformer edge deployment")
if q:
    retriever = EvidenceRetriever()
    for e in retriever.search(q, 5):
        st.write(f"**{e['id']} — {e['title']}** · score={e['score']:.3f}")
        st.caption(e["url"])
