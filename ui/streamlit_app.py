"""Chat UI. Run: streamlit run ui/streamlit_app.py  (API must be running)."""

import os
import uuid

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Ops SOP Assistant", page_icon="✈️", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:12]
    st.session_state.messages = []  # {role, content, citations?, query_id?}

chat_tab, admin_tab = st.tabs(["Ask", "Admin"])

# ---------------- Chat ----------------
with chat_tab:
    st.title("Ops SOP Assistant")
    st.caption("Answers come only from the uploaded documents, with page citations.")

    def show_citations(citations):
        for c in citations:
            with st.expander(f"[{c['n']}] {c['title']} — page {c['page']}"):
                if c.get("section_heading"):
                    st.markdown(f"**{c['section_heading']}**")
                st.write(c["snippet"] + "…")

    for i, m in enumerate(st.session_state.messages):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("citations"):
                show_citations(m["citations"])
            if m.get("query_id"):
                c1, c2, _ = st.columns([1, 1, 10])
                for col, rating, label in ((c1, 1, "👍"), (c2, -1, "👎")):
                    if col.button(label, key=f"fb{rating}{i}"):
                        requests.post(
                            f"{API_URL}/api/feedback", json={"query_id": m["query_id"], "rating": rating}, timeout=10
                        )
                        st.toast("Thanks for the feedback")

    if question := st.chat_input("e.g. What is the procedure after a bird strike?"):
        st.session_state.messages.append({"role": "user", "content": question})
        msgs = st.session_state.messages
        history = [
            {"question": msgs[j]["content"], "answer": msgs[j + 1]["content"]}
            for j in range(0, len(msgs) - 2, 2)
            if msgs[j]["role"] == "user" and msgs[j + 1]["role"] == "assistant"
        ]
        with st.spinner("Searching the documents…"):
            try:
                r = requests.post(
                    f"{API_URL}/api/chat",
                    json={"question": question, "session_id": st.session_state.session_id, "history": history},
                    timeout=60,
                )
                r.raise_for_status()
                data = r.json()
                msgs.append(
                    {
                        "role": "assistant",
                        "content": data["answer"],
                        "citations": data["citations"],
                        "query_id": data["query_id"],
                    }
                )
            except requests.RequestException as exc:
                msgs.append({"role": "assistant", "content": f"Error contacting the API: {exc}"})
        st.rerun()

# ---------------- Admin ----------------
with admin_tab:
    st.header("Documents")
    key = st.text_input("Admin key", type="password")
    headers = {"X-Admin-Key": key}

    if key:
        with st.form("upload", clear_on_submit=True):
            f = st.file_uploader("PDF", type=["pdf"])
            title = st.text_input("Title")
            version = st.text_input("Version", value="1.0")
            if st.form_submit_button("Upload and index") and f and title:
                r = requests.post(
                    f"{API_URL}/api/documents",
                    headers=headers,
                    files={"file": (f.name, f.getvalue(), "application/pdf")},
                    data={"title": title, "version": version},
                    timeout=120,
                )
                st.success("Uploaded. Indexing…") if r.ok else st.error(r.text)

        r = requests.get(f"{API_URL}/api/documents", headers=headers, timeout=10)
        if r.status_code == 401:
            st.error("Wrong admin key")
        elif r.ok:
            docs = r.json()
            st.dataframe(
                [
                    {k: d[k] for k in ("doc_id", "title", "version", "status", "pages", "chunks", "uploaded_at")}
                    for d in docs
                ],
                use_container_width=True,
            )
            for d in docs:
                cols = st.columns([4, 1, 1])
                cols[0].write(f"{d['title']} ({d['status']})" + (f" — {d['error']}" if d.get("error") else ""))
                if cols[1].button("Re-index", key=f"re{d['doc_id']}"):
                    requests.post(f"{API_URL}/api/documents/{d['doc_id']}/reindex", headers=headers, timeout=10)
                    st.rerun()
                if cols[2].button("Delete", key=f"del{d['doc_id']}"):
                    requests.delete(f"{API_URL}/api/documents/{d['doc_id']}", headers=headers, timeout=10)
                    st.rerun()

            st.header("Usage")
            m = requests.get(f"{API_URL}/api/metrics", headers=headers, timeout=10).json()
            c = st.columns(5)
            c[0].metric("Questions", m["queries"])
            c[1].metric("Grounded", f"{m['grounded_rate']:.0%}")
            c[2].metric("Avg latency", f"{m['avg_latency_ms']} ms")
            c[3].metric("👍 / 👎", f"{m['thumbs_up']} / {m['thumbs_down']}")
            c[4].metric("Cost", f"${m['total_cost_usd']:.4f}")
