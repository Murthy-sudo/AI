import time
import uuid
from io import BytesIO
from datetime import datetime

import requests
import streamlit as st

import base64

# =========================
# CONFIG
# =========================

BACKEND_URL = "http://localhost:8000"
st.set_page_config(page_title="DevopsIQ")

# =========================
# SESSION INIT
# =========================

if "sessions" not in st.session_state:
    st.session_state.sessions = {}

if "current_session" not in st.session_state:
    sid = str(uuid.uuid4())
    now = datetime.now().strftime("%I:%M %p")
    st.session_state.sessions[sid] = {
        "messages": [],
        "title": f"{now} - Client"
    }
    st.session_state.current_session = sid

if "processing" not in st.session_state:
    st.session_state.processing = False

if "dynamic_followups" not in st.session_state:
    st.session_state.dynamic_followups = []

if "role" not in st.session_state:
    st.session_state.role = "Client"

if "admin_mode" not in st.session_state:
    st.session_state.admin_mode = "Normal Chat"

if "last_role" not in st.session_state:
    st.session_state.last_role = st.session_state.role

current_messages = st.session_state.sessions[
    st.session_state.current_session
]["messages"]

# =========================
# SIDEBAR
# =========================

st.sidebar.title("⚙️ Control Panel")

role = st.sidebar.selectbox(
    "Select Role",
    ["Client", "Admin"],
    index=0 if st.session_state.role == "Client" else 1
)

# Role change → new session
if role != st.session_state.last_role:
    sid = str(uuid.uuid4())
    now = datetime.now().strftime("%I:%M %p")
    st.session_state.sessions[sid] = {
        "messages": [],
        "title": f"{now} - {role}"
    }
    st.session_state.current_session = sid
    st.session_state.dynamic_followups = []
    st.session_state.last_role = role

st.session_state.role = role

# Admin mode
if role == "Admin":
    admin_mode = st.sidebar.selectbox(
        "Admin Mode",
        ["Normal Chat", "RAG Ingest"],
        index=0 if st.session_state.admin_mode == "Normal Chat" else 1
    )
    st.session_state.admin_mode = admin_mode
else:
    admin_mode = None

st.sidebar.markdown("---")

# ➕ New Chat
if st.sidebar.button("➕ New Chat"):
    sid = str(uuid.uuid4())
    now = datetime.now().strftime("%I:%M %p")
    st.session_state.sessions[sid] = {
        "messages": [],
        "title": f"{now} - {role}"
    }
    st.session_state.current_session = sid
    st.session_state.dynamic_followups = []
    st.rerun()

st.sidebar.markdown("### 💬 Sessions")

# Session list (single clean version)
for sid, session_data in list(st.session_state.sessions.items()):

    is_active = sid == st.session_state.current_session
    container = st.sidebar.container()

    bg_color = "#4f46e5" if is_active else "#1f2937"

    container.markdown(
        f"""
        <div style="
            background-color:{bg_color};
            padding:8px;
            border-radius:8px;
            margin-bottom:6px;
            color:white;
            font-size:14px;
        ">
            {session_data["title"]}
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = container.columns([4, 1])

    if col1.button("Open", key=f"open_{sid}"):
        st.session_state.current_session = sid
        st.session_state.dynamic_followups = []
        st.rerun()

    if col2.button("🗑", key=f"delete_{sid}"):

        del st.session_state.sessions[sid]

        if sid == st.session_state.current_session:
            new_sid = str(uuid.uuid4())
            now = datetime.now().strftime("%I:%M %p")
            st.session_state.sessions[new_sid] = {
                "messages": [],
                "title": f"{now} - {role}"
            }
            st.session_state.current_session = new_sid

        st.rerun()

current_messages = st.session_state.sessions[
    st.session_state.current_session
]["messages"]

# =========================
# HEADER
# =========================

st.markdown("""
<div style="text-align:center; margin-bottom:15px;">
    <img src="https://cdn-icons-png.flaticon.com/512/4712/4712109.png" width="90">
    <div style="font-size:24px; font-weight:700; margin-top:8px;">
        DevopsIQ
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# INITIAL GREETING
# =========================

if len(current_messages) == 0:
    current_messages.append({
        "role": "assistant",
        "content": f"Hello! You are logged in as **{role}**."
    })

# =========================
# ADMIN - RAG INGEST
# =========================

if role == "Admin" and admin_mode == "RAG Ingest":

    st.info("Admin RAG Ingest Mode. Upload only. Chat disabled.")

    uploaded_file = st.file_uploader(
        "Upload document",
        type=["txt", "log", "pdf"]
    )

    if uploaded_file:
        with st.spinner("Uploading..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/upload",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type,
                        )
                    },
                )

                if response.status_code == 200:
                    st.success("✅ Ingestion successful.")
                else:
                    st.error("❌ Ingestion failed.")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

else:

    optional_file = None

    if role == "Admin" and admin_mode == "Normal Chat":
        optional_file = st.file_uploader(
            "Optional file",
            type=["txt", "log", "pdf"]
        )

    def send_message(user_input):

        st.session_state.processing = True
        st.session_state.dynamic_followups = []

        session_id = str(st.session_state.current_session)

        # Append user message FIRST
        current_messages.append({
            "role": "user",
            "content": user_input
        })

        # Auto rename after first user message
        session_data = st.session_state.sessions[st.session_state.current_session]
        if len(session_data["messages"]) == 1:
            short_title = user_input[:30] + ("..." if len(user_input) > 30 else "")
            session_data["title"] = short_title

        typing_placeholder = st.empty()
        with typing_placeholder.container():
            with st.chat_message("assistant"):
                st.markdown("✍️ _Thinking..._")

        try:
            payload = {
                "query": user_input,
                "session_id": session_id,
            }

            if role == "Admin" and admin_mode == "Normal Chat" and optional_file:
                response = requests.post(
                    f"{BACKEND_URL}/chat",
                    data=payload,
                    files={
                        "file": (
                            optional_file.name,
                            optional_file.getvalue(),
                            optional_file.type,
                        )
                    },
                )
            else:
                response = requests.post(
                    f"{BACKEND_URL}/chat",
                    data=payload
                )

            data = response.json()
            answer = data.get("answer", "No response")
            st.session_state.dynamic_followups = data.get("followups", [])

            file_bytes_base64 = data.get("file_bytes")
            file_name = data.get("file_name")

            typing_placeholder.empty()

            # Show assistant answer
            current_messages.append({
                "role": "assistant",
                "content": answer
            })

            # =========================
            # SIMPLE FILE DOWNLOAD
            # =========================

            if file_bytes_base64:
                try:
                    decoded_bytes = base64.b64decode(file_bytes_base64)
                    file_buffer = BytesIO(decoded_bytes)

                    # Fallback filename
                    if not file_name:
                        file_name = f"generated_{int(time.time())}.pdf"

                    with st.chat_message("assistant"):
                        st.download_button(
                            label="📄 Download Generated File",
                            data=file_buffer,
                            file_name=file_name,
                            mime="application/pdf"
                        )

                except Exception as e:
                    current_messages.append({
                        "role": "assistant",
                        "content": f"⚠️ File decoding failed: {str(e)}"
                    })


        except Exception as e:
            typing_placeholder.empty()
            current_messages.append({
                "role": "assistant",
                "content": f"Error: {str(e)}"
            })

        st.session_state.processing = False


    for message in current_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_prompt = st.chat_input(
        "Type your message...",
        disabled=st.session_state.processing
    )

    if user_prompt:
        send_message(user_prompt)
        st.rerun()

# =========================
# AUTO SCROLL
# =========================

st.markdown("""
<script>
window.scrollTo(0, document.body.scrollHeight);
</script>
""", unsafe_allow_html=True)
