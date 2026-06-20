import json
import streamlit as st
from backend import chatbot, llm
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from ocr_utils import scan_documents

st.set_page_config(page_title="MedVault", page_icon="🩺", layout="centered")
st.markdown("""
<style>
[data-testid="collapsedControl"] { display:none !important; }
.footer { position:fixed; bottom:0; left:0; width:100%; background:#fff; text-align:center; padding:8px; color:#888; font-size:13px; z-index:999; border-top:1px solid #ddd; }
.stBottom { margin-bottom:40px; }
.main-header { font-size:1.4rem; font-weight:600; margin-bottom:0.25rem; }
.meta { color:#666; font-size:0.8rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""<div style="text-align:center; padding:1rem 0 0.25rem 0;">
<h1 style="margin:0; font-size:1.8rem; font-weight:700;">🩺 MedVault</h1>
<p style="margin:0; color:#000; font-size:0.95rem; font-style:italic;">
see your claim through the lens of your policy</p>
</div>""", unsafe_allow_html=True)

st.markdown('<div class="footer">©copyright Owned By Ashish Sarkar | MedVault — Medical Claim Assistant </div>', unsafe_allow_html=True)

if "ocr_data" not in st.session_state:
    st.session_state.ocr_data = None
if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "message_history" not in st.session_state:
    st.session_state.message_history = []
if "tab" not in st.session_state:
    st.session_state.tab = "Process Documents"

# ── Tab buttons ──
col1, col2 = st.columns(2)
with col1:
    if st.button("📄 Process Documents", use_container_width=True):
        st.session_state.tab = "Process Documents"
        st.rerun()
with col2:
    if st.button("💬 Chat", use_container_width=True):
        st.session_state.tab = "Chat"
        st.rerun()

# ── TAB: Documents ────────────────────────────────────────────────────
if st.session_state.tab == "Process Documents":
    st.markdown('<div class="main-header">Upload Medical Documents</div>', unsafe_allow_html=True)
    st.markdown('<div class="meta">Upload one PDF/JPEG each for Prescription, Bills, and Lab Reports.</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        rx = st.file_uploader("💊 Prescription", type=["pdf", "jpeg", "jpg"], key="rx", disabled=st.session_state.submitted)
    with col2:
        bills = st.file_uploader("🧾 Bills", type=["pdf", "jpeg", "jpg"], key="bills", disabled=st.session_state.submitted)
    with col3:
        lab = st.file_uploader("🧪 Lab Reports", type=["pdf", "jpeg", "jpg"], key="lab", disabled=st.session_state.submitted)

    if st.button("Submit", type="primary", disabled=st.session_state.submitted):
        errors = []
        if rx is None:
            errors.append("Prescription is required")
        if bills is None:
            errors.append("Bills is required")
        if lab is None:
            errors.append("Lab Reports is required")

        for name, f in [("Prescription", rx), ("Bills", bills), ("Lab Reports", lab)]:
            if f is not None:
                if f.size > 500 * 1024:
                    errors.append(f"{name} exceeds 500 KB")
                ext = f.name.split(".")[-1].lower()
                if ext not in ("pdf", "jpeg", "jpg"):
                    errors.append(f"{name} must be PDF or JPEG")

        if errors:
            for e in errors:
                st.error(e)
        else:
            with st.spinner("Scanning documents …"):
                result = scan_documents(rx, bills, lab)

            if not isinstance(result, dict):
                st.error(f"⚠️ scan_documents returned: {type(result).__name__} = {repr(result)[:200]}")
            elif "error" in result:
                st.error(f"❌ {result['error']}")
            else:
                st.session_state.ocr_data = result
                st.session_state.submitted = True
                st.session_state.message_history = []
                st.success("Documents processed successfully!")
                st.info("Switch to the **💬 Chat** tab to ask about your claim.")

    if st.session_state.submitted and st.session_state.ocr_data:
        st.success("✅ Documents processed")
        if st.button("Reset", type="secondary"):
            st.session_state.ocr_data = None
            st.session_state.submitted = False
            st.session_state.message_history = []
            st.rerun()

# ── TAB: Chat ─────────────────────────────────────────────────────────
elif st.session_state.tab == "Chat":
    if not st.session_state.ocr_data:
        st.info("📄 No documents processed yet. Go to the **Process Documents** tab first.")
    else:
        data = st.session_state.ocr_data
        st.markdown(f'<div class="meta">🧑 {data.get("Patient_name", "—")} &nbsp;|&nbsp; '
                    f'🩺 {data.get("Diagnosis", "—")} &nbsp;|&nbsp; '
                    f'💰 ${data.get("Bill_amount", "—")}</div>',
                    unsafe_allow_html=True)

        for msg in st.session_state.message_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_input = st.chat_input("Ask about your claim …")

        if user_input:
            st.session_state.message_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            from retrieval import get_context
            policy_ctx = get_context(user_input)
            system_text = (
                f"You are a medical claim assistant.\n\n"
                f"Claim Data (from OCR):\n{json.dumps(st.session_state.ocr_data, indent=2)}\n\n"
            )
            if policy_ctx:
                system_text += (
                    f"Relevant Policy Clauses (from insurance document):\n{policy_ctx}\n\n"
                )
            system_text += "Answer based on the claim data and policy clauses above. Be concise."
            system = SystemMessage(content=system_text)
            chat_messages = [system]
            for m in st.session_state.message_history[:-1][-8:]:
                chat_messages.append(
                    HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
                )
            chat_messages.append(HumanMessage(content=user_input))
            response = chatbot.invoke({"messages": chat_messages}, config={"configurable": {"thread_id": "thread-1"}})
            reply = response["messages"][-1].content

            st.session_state.message_history.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)