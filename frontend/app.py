import streamlit as st
import requests
import pandas as pd

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Scientific RAG",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔬 Multi-Document Scientific RAG")
st.caption("Hybrid retrieval · Citation-aware · Meta-analysis")

# ----- sidebar -----
with st.sidebar:
    st.header("Configuration")

    mode = st.selectbox("Query mode", ["auto", "single", "multi"])
    top_k = st.slider("Chunks to retrieve", 3, 10, 5)

    # load papers for single-paper mode
    try:
        papers_resp = requests.get(f"{API_BASE}/papers", timeout=5)
        papers = papers_resp.json()
    except Exception:
        papers = []
        st.warning("API not reachable. Start run_server.py first.")

    paper_options = {f"{p['title'] or p['paper_id']} ({p['year']})": p["paper_id"]
                     for p in papers}
    selected_title = st.selectbox(
        "Paper (single mode only)",
        ["All papers"] + list(paper_options.keys())
    )
    paper_id = paper_options.get(selected_title) if selected_title != "All papers" else None

    st.divider()
    st.header("Venue Editor")
    if papers:
        edit_title = st.selectbox("Paper", list(paper_options.keys()), key="venue_edit")
        venue_choice = st.selectbox("Venue tier", ["A*", "A", "B", "Workshop", "Preprint", "Unknown"])
        if st.button("Update venue"):
            pid = paper_options[edit_title]
            r = requests.post(
                f"{API_BASE}/papers/venue",
                json={"paper_id": pid, "venue": venue_choice}
            )
            st.success("Updated." if r.ok else "Failed.")


# ----- main: query panel -----
question = st.text_input("Ask a question about your papers", placeholder="e.g. What dataset gives the highest F1 for BERT?")

col_ask, col_clear = st.columns([1, 6])
with col_ask:
    ask_clicked = st.button("Ask", type="primary", width="stretch")

if ask_clicked and question.strip():
    payload = {
        "question": question,
        "mode":     mode,
        "paper_id": paper_id if mode == "single" else None,
        "top_k":    top_k,
    }

    with st.spinner("Retrieving and generating..."):
        try:
            resp = requests.post(
                f"{API_BASE}/query",
                json=payload,
                timeout=60,
            )
            if not resp.ok:
                st.error(f"API error {resp.status_code}: {resp.text}")
                st.stop()
            data = resp.json()
        except requests.exceptions.Timeout:
            st.error("Request timed out. The LLM may be slow. Try again.")
            st.stop()
        except Exception as e:
            st.error(f"Request failed: {e}")
            st.stop()

    # ----- answer -----
    st.markdown("### Answer")
    st.markdown(data.get("answer", "No answer returned."))

    # ----- retrieved chunks -----
    with st.expander(f"Retrieved chunks ({len(data.get('retrieved_chunks', []))})"):
        for chunk in data.get("retrieved_chunks", []):
            st.markdown(
                f"**{chunk['paper_id']}** | `{chunk['section']}` | score: `{chunk['score']}`"
            )
            st.caption(chunk["text"][:300] + ("..." if len(chunk["text"]) > 300 else ""))
            st.divider()

    # ----- consensus table -----
    if "consensus_table" in data and data["consensus_table"]:
        st.markdown("### Consensus Table")
        table_rows = data["consensus_table"]
        df = pd.DataFrame([
            {
                "Metric":         r["metric"],
                "Dataset":        r["dataset"],
                "W.Mean":         r.get("weighted_mean"),
                "S.Mean":         r.get("simple_mean"),
                "Std":            r.get("std"),
                "N":              r.get("n"),
                "Confidence":     r.get("confidence"),
                "Contradiction":  "⚠️ YES" if r.get("contradiction") else "ok",
                "Papers":         ", ".join(r.get("papers", [])),
            }
            for r in table_rows
        ])

        def highlight_contradiction(row):
            if row["Contradiction"] == "⚠️ YES":
                return ["background-color: #fff3cd"] * len(row)
            return [""] * len(row)

        st.dataframe(
            df.style.apply(highlight_contradiction, axis=1),
            width="stretch",
            hide_index=True,
        )

elif ask_clicked and not question.strip():
    st.warning("Please enter a question.")


# ----- footer: extraction browser -----
with st.expander("Browse extracted results"):
    try:
        ext_resp = requests.get(f"{API_BASE}/extractions?limit=100", timeout=5)
        ext_data = ext_resp.json()
        if ext_data:
            st.dataframe(pd.DataFrame(ext_data), width="stretch")
        else:
            st.info("No extractions yet. Run ingestion with extraction enabled.")
    except Exception:
        st.info("API not reachable.")

