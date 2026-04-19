import json
import os
import re
import uuid
from datetime import datetime
from io import BytesIO
from pathlib import Path

# Must be set before importing chromadb to avoid protobuf C-extension incompatibility.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PYDANTIC_DISABLE_PLUGINS", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

from demo_embeddings import OpenAIEmbeddingFunction


st.set_page_config(page_title="AI Interview Analysis Assistant", layout="wide")


DB_DIR = "db/chroma_demo"
COLLECTION_NAME = "demo_interviews"
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR = LOG_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
SESSION_INDEX_PATH = LOG_DIR / "session_index.json"
DEFAULT_DEMO_PASSWORD = "demo2026"
DEFAULT_CHAT_MODEL = "gpt-4o"

SITE_FILTER_OPTIONS = {
    "All": None,
    "Site A": "Site A",
    "Site B": "Site B",
    "Site C": "Site C",
}
COLLAR_FILTER_OPTIONS = {
    "All": None,
    "White-collar": "white",
    "Blue-collar": "blue",
}
ROLE_FILTER_OPTIONS = {
    "All": None,
    "Manager": "manager",
    "Employee": "employee",
}
EXAMPLE_QUESTIONS = [
    "What do blue-collar employees say about scheduling?",
    "Compare managers and employees on communication.",
    "Which sites describe the strongest team collaboration?",
]


DEFAULT_TASK_PROMPT = """TASK: Answer the user's question using only the provided synthetic interview excerpts.
If the available excerpts are insufficient or contradictory, state that clearly.

OUTPUT FORMAT:
Findings: provide 4-6 concise bullets. Each bullet should include a claim and a short explanation.
Evidence: for each finding, list the site, interviewee number, collar type, role, question number, and 1-2 short quotes of at most 25 words each.

RULES:
Do not use outside knowledge. Do not invent facts.
When you generalize, state how many interviewees or excerpts support the claim."""

DEFAULT_SYSTEM_PROMPT = (
    "ROLE: You are a helpful organizational research analyst specializing in qualitative "
    f"workplace culture analysis. {DEFAULT_TASK_PROMPT}"
)

DEFAULT_REMINDER_PROMPT = (
    "Reminder: You are a helpful organizational research analyst. Continue the multi-turn "
    f"conversation using only the retrieved synthetic interview sources. {DEFAULT_TASK_PROMPT}"
)


def get_secret_or_env(name: str, default: str = "") -> str:
    return st.secrets.get(name, os.getenv(name, default))


def check_password() -> bool:
    expected_password = st.secrets.get("app_password", DEFAULT_DEMO_PASSWORD)

    def password_entered():
        if st.session_state["password"] == expected_password:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.text_input(
        "Enter the demo password:",
        type="password",
        on_change=password_entered,
        key="password",
    )

    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("Incorrect password. Please try again.")

    st.caption("Shared portfolio demo password: `demo2026`")
    return False


def load_example_question(question: str) -> None:
    st.session_state["user_query_input"] = question


def toggle_sources_menu(expanded_key: str) -> None:
    st.session_state[expanded_key] = not st.session_state.get(expanded_key, True)


if not check_password():
    st.stop()


components.html(
    """
<script>
window.onbeforeunload = function () {
  return "Leave this page? Your conversation is saved, but unsaved edits may be lost.";
};
</script>
""",
    height=0,
)


def get_or_create_session_id() -> str:
    sid = st.query_params.get("sid")
    if not sid:
        sid = str(uuid.uuid4())
        st.query_params["sid"] = sid
    return sid


def session_log_path(session_id: str) -> Path:
    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", session_id)
    return LOG_DIR / f"session_{safe_id}.json"


def sanitize_filename_part(value: str, fallback: str = "checkpoint") -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", value.strip())
    return cleaned or fallback


def checkpoint_log_path(session_id: str, session_name: str) -> Path:
    safe_id = sanitize_filename_part(session_id, fallback="session")
    safe_name = sanitize_filename_part(session_name, fallback="unnamed")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return CHECKPOINT_DIR / f"{timestamp}_{safe_name}_{safe_id}.json"


def load_session_index() -> list[dict]:
    if not SESSION_INDEX_PATH.exists():
        return []

    try:
        data = json.loads(SESSION_INDEX_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def save_session_index(index_entries: list[dict]):
    SESSION_INDEX_PATH.write_text(json.dumps(index_entries, ensure_ascii=False, indent=2), encoding="utf-8")


def append_checkpoint_to_index(checkpoint_path: Path, source_session_id: str, session_name: str, saved_at: str):
    index_entries = load_session_index()
    index_entries.append(
        {
            "checkpoint_file": checkpoint_path.name,
            "source_session_id": source_session_id,
            "session_name": session_name,
            "saved_at": saved_at,
        }
    )
    save_session_index(index_entries)


def remove_checkpoint_from_index(checkpoint_filename: str):
    index_entries = load_session_index()
    filtered_entries = [entry for entry in index_entries if entry.get("checkpoint_file") != checkpoint_filename]
    save_session_index(filtered_entries)


def format_checkpoint_label(index_entry: dict) -> str:
    session_name = index_entry.get("session_name", "Untitled checkpoint")
    saved_at = index_entry.get("saved_at", "")
    source_session_id = index_entry.get("source_session_id", "unknown")
    return f"{session_name} ({saved_at}) - source: {source_session_id}"


def save_session_data(session_id: str):
    data = {
        "session_id": session_id,
        "updated_at": datetime.now().isoformat(),
        "chat_messages": st.session_state.get("chat_messages", []),
        "sources_by_turn": st.session_state.get("sources_by_turn", []),
        "editable_log": st.session_state.get("editable_log", ""),
        "log_msg_count": st.session_state.get("log_msg_count", 0),
        "log_source_count": st.session_state.get("log_source_count", 0),
        "system_prompt": st.session_state.get("system_prompt", ""),
        "turn_reminder_prompt": st.session_state.get("turn_reminder_prompt", ""),
        "system_prompt_sent": st.session_state.get("system_prompt_sent", False),
    }
    session_log_path(session_id).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_session_data(session_id: str):
    path = session_log_path(session_id)
    if not path.exists():
        return

    data = json.loads(path.read_text(encoding="utf-8"))
    st.session_state["chat_messages"] = data.get("chat_messages", [])
    st.session_state["sources_by_turn"] = data.get("sources_by_turn", [])
    st.session_state["editable_log"] = data.get("editable_log", "")
    st.session_state["system_prompt"] = data.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    st.session_state["turn_reminder_prompt"] = data.get("turn_reminder_prompt", DEFAULT_REMINDER_PROMPT)
    st.session_state["system_prompt_sent"] = data.get("system_prompt_sent", False)

    if st.session_state["editable_log"] and "log_msg_count" not in data:
        st.session_state["log_msg_count"] = len(st.session_state["chat_messages"])
        st.session_state["log_source_count"] = len(st.session_state["sources_by_turn"])
    else:
        st.session_state["log_msg_count"] = data.get("log_msg_count", 0)
        st.session_state["log_source_count"] = data.get("log_source_count", 0)


def extract_interviewee_no(meta: dict) -> str:
    if meta.get("interviewee_no") is not None:
        return str(meta["interviewee_no"])
    return "unknown"


def extract_answer_text(doc: str) -> str:
    for marker in ("ANSWER:", "Answer:"):
        if marker in doc:
            return doc.split(marker, 1)[1].strip()
    return doc.strip()


def format_persona(collar: str, role: str) -> str:
    collar_label = "white-collar" if collar == "white" else "blue-collar" if collar == "blue" else "unknown collar"
    role_label = role if role else "unknown role"
    return f"{collar_label} {role_label}"


def format_log_entries(chat_messages, sources_by_turn, start_msg_index=0, start_source_index=0) -> str:
    lines = []

    if start_msg_index > 0 or start_source_index > 0:
        lines.append(f"\n--- UPDATE ({datetime.now().strftime('%H:%M:%S')}) ---\n")

    if start_msg_index < len(chat_messages):
        if start_msg_index == 0:
            lines.append("# AI Interview Analysis Assistant - conversation log\n")

        for i, msg in enumerate(chat_messages[start_msg_index:], start=start_msg_index + 1):
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{i}. {role}:")
            lines.append(msg["content"])
            lines.append("")

    if start_source_index < len(sources_by_turn):
        lines.append("\n# Sources by turn")
        for i, sources in enumerate(sources_by_turn[start_source_index:], start=start_source_index + 1):
            lines.append(f"\n## Turn {i}")
            for src in sources:
                persona = format_persona(src.get("collar"), src.get("role"))
                lines.append(
                    f"- {src['site']} | interviewee #{src['interviewee_no']} | {persona} | "
                    f"question #{src['question_no']} | answer: {src['answer_text']}"
                )

    return "\n".join(lines)


def generate_full_log_text(chat_messages, sources_by_turn) -> str:
    return format_log_entries(chat_messages, sources_by_turn, start_msg_index=0, start_source_index=0)


def build_checkpoint_text(session_id: str) -> str:
    log_body = st.session_state.get("editable_log", "").strip()
    if not log_body:
        log_body = generate_full_log_text(
            st.session_state.get("chat_messages", []), st.session_state.get("sources_by_turn", [])
        )

    return f"session_id: {session_id}\n\n{log_body}\n"


def build_checkpoint_payload(session_id: str, session_name: str, saved_at: str) -> dict:
    log_body = st.session_state.get("editable_log", "").strip()
    if not log_body:
        log_body = generate_full_log_text(
            st.session_state.get("chat_messages", []), st.session_state.get("sources_by_turn", [])
        )

    return {
        "session_id": session_id,
        "session_name": session_name,
        "saved_at": saved_at,
        "chat_messages": st.session_state.get("chat_messages", []),
        "sources_by_turn": st.session_state.get("sources_by_turn", []),
        "editable_log": log_body,
        "log_msg_count": st.session_state.get("log_msg_count", 0),
        "log_source_count": st.session_state.get("log_source_count", 0),
        "system_prompt": st.session_state.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
        "turn_reminder_prompt": st.session_state.get("turn_reminder_prompt", DEFAULT_REMINDER_PROMPT),
        "system_prompt_sent": st.session_state.get("system_prompt_sent", False),
        "legacy_markdown": build_checkpoint_text(session_id=session_id),
    }


def load_checkpoint_payload(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("The checkpoint file format is invalid.")
    return data


def save_session_checkpoint(session_id: str, session_name: str) -> Path:
    saved_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path = checkpoint_log_path(session_id=session_id, session_name=session_name)
    checkpoint_payload = build_checkpoint_payload(session_id=session_id, session_name=session_name, saved_at=saved_at)
    path.write_text(json.dumps(checkpoint_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    append_checkpoint_to_index(
        checkpoint_path=path,
        source_session_id=session_id,
        session_name=session_name,
        saved_at=saved_at,
    )
    return path


def get_checkpoint_files_for_current_session(session_id: str):
    indexed_checkpoints = []
    for entry in load_session_index():
        if entry.get("source_session_id") != session_id:
            continue
        checkpoint_path = CHECKPOINT_DIR / entry.get("checkpoint_file", "")
        if checkpoint_path.exists():
            indexed_checkpoints.append({"path": checkpoint_path, "entry": entry})

    return sorted(indexed_checkpoints, key=lambda item: item["entry"].get("saved_at", ""), reverse=True)


def get_all_checkpoints():
    indexed_checkpoints = []
    for entry in load_session_index():
        checkpoint_path = CHECKPOINT_DIR / entry.get("checkpoint_file", "")
        if checkpoint_path.exists():
            indexed_checkpoints.append({"path": checkpoint_path, "entry": entry})

    return sorted(indexed_checkpoints, key=lambda item: item["entry"].get("saved_at", ""), reverse=True)


def apply_checkpoint_to_current_session(checkpoint_payload: dict):
    st.session_state["chat_messages"] = checkpoint_payload.get("chat_messages", [])
    st.session_state["sources_by_turn"] = checkpoint_payload.get("sources_by_turn", [])
    st.session_state["editable_log"] = checkpoint_payload.get("editable_log", "")

    if "log_msg_count" in checkpoint_payload:
        st.session_state["log_msg_count"] = checkpoint_payload["log_msg_count"]
        st.session_state["log_source_count"] = checkpoint_payload["log_source_count"]
    else:
        st.session_state["log_msg_count"] = len(st.session_state["chat_messages"])
        st.session_state["log_source_count"] = len(st.session_state["sources_by_turn"])

    st.session_state["system_prompt"] = checkpoint_payload.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    st.session_state["turn_reminder_prompt"] = checkpoint_payload.get("turn_reminder_prompt", DEFAULT_REMINDER_PROMPT)
    st.session_state["system_prompt_sent"] = checkpoint_payload.get("system_prompt_sent", False)


def delete_checkpoint(checkpoint_path: Path):
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    remove_checkpoint_from_index(checkpoint_path.name)


def _load_pdf_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]

    wrapped = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        if draw.textlength(trial, font=font) <= max_width:
            current = trial
        else:
            wrapped.append(current)
            current = word
    wrapped.append(current)
    return wrapped


def generate_log_pdf(log_text: str) -> bytes:
    page_width, page_height = 1240, 1754
    margin_x = 72
    margin_top = 72
    margin_bottom = 72
    max_text_width = page_width - (2 * margin_x)

    title_font = _load_pdf_font(42, bold=True)
    heading_font = _load_pdf_font(30, bold=True)
    body_font = _load_pdf_font(24, bold=False)

    def line_height(font: ImageFont.ImageFont, ratio: float = 1.35) -> int:
        ascent, descent = font.getmetrics()
        return int((ascent + descent) * ratio)

    title_lh = line_height(title_font)
    heading_lh = line_height(heading_font)
    body_lh = line_height(body_font)

    def new_page():
        page = Image.new("RGB", (page_width, page_height), "white")
        return page, ImageDraw.Draw(page), margin_top

    pages = []
    page, draw, y = new_page()

    for raw_line in (log_text.splitlines() or [""]):
        stripped = raw_line.strip()
        if not stripped:
            y += body_lh // 2
            continue

        text = stripped
        font = body_font
        lh = body_lh

        if stripped.startswith("# "):
            text = stripped[2:].strip()
            font = title_font
            lh = title_lh
        elif stripped.startswith("## "):
            text = stripped[3:].strip()
            font = heading_font
            lh = heading_lh
        elif re.match(r"^\d+\.\sUser:$", stripped):
            q_no = stripped.split(".", 1)[0]
            text = f"Question {q_no}"
            font = heading_font
            lh = heading_lh
        elif re.match(r"^\d+\.\sAssistant:$", stripped):
            text = "Answer"
            font = heading_font
            lh = heading_lh

        wrapped_lines = _wrap_text(draw, text, font, max_text_width)
        block_height = len(wrapped_lines) * lh + (body_lh // 3)
        if y + block_height > page_height - margin_bottom:
            pages.append(page)
            page, draw, y = new_page()

        for line in wrapped_lines:
            draw.text((margin_x, y), line, font=font, fill="black")
            y += lh

        y += body_lh // 3

    pages.append(page)

    buffer = BytesIO()
    pages[0].save(buffer, format="PDF", resolution=150.0, save_all=True, append_images=pages[1:])
    return buffer.getvalue()


def build_where_clause(site_label: str, question_label: str, collar_label: str, role_label: str):
    filters = []
    site_value = SITE_FILTER_OPTIONS.get(site_label)
    collar_value = COLLAR_FILTER_OPTIONS.get(collar_label)
    role_value = ROLE_FILTER_OPTIONS.get(role_label)

    if site_value is not None:
        filters.append({"site": site_value})
    if question_label != "All":
        filters.append({"question_no": int(question_label)})
    if collar_value is not None:
        filters.append({"collar": collar_value})
    if role_value is not None:
        filters.append({"role": role_value})

    if not filters:
        return None
    if len(filters) == 1:
        return filters[0]
    return {"$and": filters}


def init_state_once():
    if "session_loaded" in st.session_state:
        return

    sid = get_or_create_session_id()
    st.session_state["session_id"] = sid
    st.session_state.setdefault("chat_messages", [])
    st.session_state.setdefault("sources_by_turn", [])
    st.session_state.setdefault("editable_log", "")
    st.session_state.setdefault("log_msg_count", 0)
    st.session_state.setdefault("log_source_count", 0)
    st.session_state.setdefault("system_prompt", DEFAULT_SYSTEM_PROMPT)
    st.session_state.setdefault("turn_reminder_prompt", DEFAULT_REMINDER_PROMPT)
    st.session_state.setdefault("system_prompt_sent", False)

    load_session_data(sid)
    st.session_state["session_loaded"] = True


init_state_once()


st.sidebar.header("Settings")
st.sidebar.caption("Synthetic dataset. Fictional company and sites.")

st.sidebar.subheader("Model instructions")
edited_system_prompt = st.sidebar.text_area(
    "System message",
    value=st.session_state.get("system_prompt", ""),
    height=160,
    help="Sent once at the beginning of a session conversation.",
)
if edited_system_prompt != st.session_state.get("system_prompt", ""):
    st.session_state["system_prompt"] = edited_system_prompt
    st.session_state["system_prompt_sent"] = False
    if st.session_state.get("session_id"):
        save_session_data(st.session_state["session_id"])

st.sidebar.subheader("Per-turn reminder")
edited_turn_reminder = st.sidebar.text_area(
    "Fixed message before every new question",
    value=st.session_state.get("turn_reminder_prompt", ""),
    height=120,
    help="Sent as a user message before each new question.",
)
if edited_turn_reminder != st.session_state.get("turn_reminder_prompt", ""):
    st.session_state["turn_reminder_prompt"] = edited_turn_reminder
    if st.session_state.get("session_id"):
        save_session_data(st.session_state["session_id"])

api_key = get_secret_or_env("OPENAI_API_KEY")
if not api_key:
    api_key = st.sidebar.text_input("OpenAI API key", type="password")

chat_model = get_secret_or_env("OPENAI_CHAT_MODEL", DEFAULT_CHAT_MODEL)

if not api_key:
    st.warning("Add an OpenAI API key in the sidebar to continue.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.subheader("Session checkpoints")
session_checkpoints = get_checkpoint_files_for_current_session(st.session_state["session_id"])
if not session_checkpoints:
    st.sidebar.caption("No checkpoints saved in this session.")
else:
    selected_checkpoint_local = st.sidebar.selectbox(
        "Saved logs for this session",
        options=session_checkpoints,
        format_func=lambda item: format_checkpoint_label(item["entry"]),
        key="selected_checkpoint_local",
    )

    if st.sidebar.button("Load current-session checkpoint"):
        checkpoint_payload = load_checkpoint_payload(selected_checkpoint_local["path"])
        source_session_id = checkpoint_payload.get("session_id", "unknown")
        apply_checkpoint_to_current_session(checkpoint_payload)
        st.session_state["loaded_from_session_id"] = source_session_id
        save_session_data(st.session_state["session_id"])
        st.rerun()

    st.sidebar.download_button(
        label="Download selected checkpoint",
        data=selected_checkpoint_local["path"].read_text(encoding="utf-8"),
        file_name=selected_checkpoint_local["path"].name,
        mime="application/json",
    )

st.sidebar.subheader("All saved checkpoints")
all_checkpoints = get_all_checkpoints()
if not all_checkpoints:
    st.sidebar.caption("No indexed checkpoints yet.")
else:
    selected_checkpoint_global = st.sidebar.selectbox(
        "Saved logs across all sessions",
        options=all_checkpoints,
        format_func=lambda item: format_checkpoint_label(item["entry"]),
        key="selected_checkpoint_global",
    )

    if st.sidebar.button("Load selected checkpoint"):
        checkpoint_payload = load_checkpoint_payload(selected_checkpoint_global["path"])
        source_session_id = checkpoint_payload.get("session_id", "unknown")
        apply_checkpoint_to_current_session(checkpoint_payload)
        st.session_state["loaded_from_session_id"] = source_session_id
        save_session_data(st.session_state["session_id"])
        st.rerun()

    if st.sidebar.button("Delete selected checkpoint"):
        st.session_state["checkpoint_to_delete"] = selected_checkpoint_global["path"].name
        st.rerun()


@st.dialog("Delete checkpoint")
def delete_checkpoint_dialog():
    checkpoint_name = st.session_state.get("checkpoint_to_delete")
    if not checkpoint_name:
        st.rerun()

    st.warning(f"Delete this checkpoint?\n\n`{checkpoint_name}`")
    delete_col, cancel_col = st.columns(2)

    with delete_col:
        if st.button("Yes, delete", type="primary", key="confirm_delete_checkpoint"):
            checkpoint_path = CHECKPOINT_DIR / checkpoint_name
            delete_checkpoint(checkpoint_path)
            st.session_state["deleted_checkpoint_name"] = checkpoint_name
            st.session_state.pop("checkpoint_to_delete", None)
            st.rerun()

    with cancel_col:
        if st.button("Cancel", key="cancel_delete_checkpoint"):
            st.session_state.pop("checkpoint_to_delete", None)
            st.rerun()


if st.session_state.get("checkpoint_to_delete"):
    delete_checkpoint_dialog()

if st.session_state.get("deleted_checkpoint_name"):
    st.sidebar.success(f"Deleted: `{st.session_state['deleted_checkpoint_name']}`")
    del st.session_state["deleted_checkpoint_name"]


@st.cache_resource
def get_db_collection(client_api_key: str):
    import chromadb

    ef = OpenAIEmbeddingFunction(
        api_key=client_api_key,
        model_name="text-embedding-3-large",
    )
    if not os.path.exists(DB_DIR):
        raise FileNotFoundError(f"The database folder '{DB_DIR}' was not found. Run scripts/rebuild_chroma.py first.")

    settings = chromadb.Settings(anonymized_telemetry=False)
    client = chromadb.PersistentClient(path=DB_DIR, settings=settings)
    return client.get_collection(name=COLLECTION_NAME, embedding_function=ef)


try:
    collection = get_db_collection(api_key)
    llm_client = OpenAI(api_key=api_key)
except Exception as e:
    st.error(f"Could not connect to the interview database: {e}")
    st.stop()


st.title("AI Interview Analysis Assistant")
st.caption(f"Active session ID: `{st.session_state['session_id']}`")
loaded_from_session_id = st.session_state.get("loaded_from_session_id")
if loaded_from_session_id:
    if loaded_from_session_id != st.session_state["session_id"]:
        st.warning(f"A checkpoint from another session was loaded. Source session ID: `{loaded_from_session_id}`")
    else:
        st.info(f"Loaded checkpoint source: `{loaded_from_session_id}`")

st.markdown(
    "**Ask questions about a fictional workplace interview dataset for Hungarian Coal Mining Industrial Association. "
    "All respondents, sites, and answers are synthetic.**"
)

st.markdown(
    """
    <style>
    div[data-testid="stForm"] {
        position: sticky;
        top: 1.25rem;
        z-index: 20;
        background: var(--secondary-background-color);
        border: 1px solid color-mix(in srgb, var(--text-color) 12%, transparent);
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 6px 24px color-mix(in srgb, var(--text-color) 10%, transparent);
        backdrop-filter: blur(6px);
        margin: 1rem auto 1.5rem auto;
        max-width: 1100px;
    }

    div[data-testid="stForm"] [data-testid="stFormSubmitButton"] {
        display: flex;
        justify-content: flex-end;
        align-items: flex-end;
        height: 100%;
    }

    div[data-testid="stForm"] [data-testid="stHorizontalBlock"] [data-testid="stFormSubmitButton"] button {
        font-size: 0.78rem;
        line-height: 1.15;
        min-height: 2.25rem;
        padding: 0.3rem 0.45rem;
    }

    .send-button-spacer {
        height: 1.75rem;
    }

    button[kind="primaryFormSubmit"],
    button[data-testid="stBaseButton-primaryFormSubmit"] {
        background: #111827;
        border-color: #111827;
        color: #ffffff;
        align-items: center;
        display: inline-flex;
        font-size: 1.15rem;
        justify-content: center;
        line-height: 1;
        min-width: 3rem;
        text-align: center;
    }

    button[kind="primaryFormSubmit"]:hover,
    button[data-testid="stBaseButton-primaryFormSubmit"]:hover {
        background: #020617;
        border-color: #020617;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if st.sidebar.button("Log out"):
    st.session_state["password_correct"] = False
    st.rerun()


def render_sources_menu(sources_data, menu_id):
    expanded_key = f"sources_expanded_{menu_id}"
    st.session_state.setdefault(expanded_key, True)
    arrow = "▲" if st.session_state[expanded_key] else "▼"
    st.button(
        f"Retrieved sources ({len(sources_data)}) {arrow}",
        key=f"sources_toggle_{menu_id}",
        on_click=toggle_sources_menu,
        args=(expanded_key,),
        use_container_width=True,
    )

    if st.session_state[expanded_key]:
        for idx, src in enumerate(sources_data, start=1):
            with st.container(border=True):
                st.markdown(f"#### Source {idx}")
                meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
                meta_col1.markdown(f"**Site:** {src['site']}")
                meta_col2.markdown(f"**Interviewee:** #{src['interviewee_no']}")
                meta_col3.markdown(f"**Persona:** {format_persona(src.get('collar'), src.get('role'))}")
                meta_col4.markdown(f"**Question:** #{src['question_no']}")
                st.markdown("**Answer:**")
                st.code(src["answer_text"], language=None)


assistant_msg_index = 0
for msg in st.session_state["chat_messages"]:
    role = msg["role"]
    with st.chat_message("assistant" if role == "assistant" else "user"):
        st.markdown(msg["content"])

        if role == "assistant":
            if "sources_by_turn" in st.session_state and len(st.session_state["sources_by_turn"]) > assistant_msg_index:
                sources = st.session_state["sources_by_turn"][assistant_msg_index]
                if sources:
                    render_sources_menu(sources, assistant_msg_index)
            assistant_msg_index += 1


with st.form("question_form", clear_on_submit=False):
    st.subheader("Question")
    user_query = st.text_input(
        "Type your question",
        placeholder=EXAMPLE_QUESTIONS[0],
        key="user_query_input",
    ).strip()

    st.caption("Examples")
    example_cols = st.columns(3)
    for index, example_question in enumerate(EXAMPLE_QUESTIONS, start=1):
        with example_cols[index - 1]:
            st.form_submit_button(
                f"{index}. {example_question}",
                key=f"example_question_{index}",
                on_click=load_example_question,
                args=(example_question,),
                use_container_width=True,
            )

    filter_col1, filter_col2, filter_col3, filter_col4, filter_col5, submit_col = st.columns(
        [1.0, 0.8, 1.0, 0.9, 0.7, 0.32]
    )

    with filter_col1:
        site_filter = st.selectbox("Site", list(SITE_FILTER_OPTIONS.keys()), key="site_filter")

    with filter_col2:
        q_options = ["All"] + [str(i) for i in range(1, 11)]
        question_filter = st.selectbox("Question", q_options, key="question_filter")

    with filter_col3:
        collar_filter = st.selectbox("Collar type", list(COLLAR_FILTER_OPTIONS.keys()), key="collar_filter")

    with filter_col4:
        role_filter = st.selectbox("Role", list(ROLE_FILTER_OPTIONS.keys()), key="role_filter")

    with filter_col5:
        top_k = st.number_input("Sources", min_value=1, max_value=20, value=5, step=1, key="top_k")

    with submit_col:
        st.markdown('<div class="send-button-spacer"></div>', unsafe_allow_html=True)
        submit_query = st.form_submit_button("↑", type="primary", use_container_width=True)

    st.caption(
        "Each answer uses the most recent chat history, the retrieved interview excerpts, the session system "
        "message, and the per-turn reminder."
    )

if submit_query and not user_query.strip():
    st.warning("Please enter a question.")

if submit_query and user_query.strip():
    user_query = user_query.strip()
    st.session_state["chat_messages"].append({"role": "user", "content": user_query})

    where_clause = build_where_clause(site_filter, question_filter, collar_filter, role_filter)

    with st.spinner("Searching the interview database and drafting an answer..."):
        try:
            results = collection.query(query_texts=[user_query], n_results=int(top_k), where=where_clause)

            docs = results["documents"][0] if results["documents"] else []
            metas = results["metadatas"][0] if results["metadatas"] else []

            if not docs:
                assistant_text = "No relevant answer was found with the selected filters."
                sources_data = []
            else:
                context_str = ""
                sources_data = []

                for doc, meta in zip(docs, metas):
                    site = meta.get("site", "Unknown")
                    q_no = meta.get("question_no", "?")
                    interviewee_no = extract_interviewee_no(meta)
                    collar = meta.get("collar", "unknown")
                    role = meta.get("role", "unknown")
                    answer_text = extract_answer_text(doc)

                    context_str += (
                        f"\nSOURCE: {site} (question {q_no}, interviewee #{interviewee_no}, "
                        f"{format_persona(collar, role)})\n{doc}\n"
                    )
                    sources_data.append(
                        {
                            "site": site,
                            "question_no": q_no,
                            "interviewee_no": interviewee_no,
                            "collar": collar,
                            "role": role,
                            "answer_text": answer_text,
                        }
                    )

                history_for_llm = []
                for m in st.session_state["chat_messages"][-8:-1]:
                    history_for_llm.append({"role": m["role"], "content": m["content"]})

                system_prompt = st.session_state.get("system_prompt", "")
                turn_reminder_prompt = st.session_state.get("turn_reminder_prompt", "")
                user_msg = (
                    f"Current question: {user_query}\n"
                    f"Filters: site={site_filter}, question={question_filter}, "
                    f"collar={collar_filter}, role={role_filter}\n\n"
                    f"Context:\n{context_str}"
                )

                llm_messages = []
                if not st.session_state.get("system_prompt_sent", False) and system_prompt.strip():
                    llm_messages.append({"role": "system", "content": system_prompt})

                llm_messages.extend(history_for_llm)

                if turn_reminder_prompt.strip():
                    llm_messages.append({"role": "user", "content": turn_reminder_prompt})

                llm_messages.append({"role": "user", "content": user_msg})

                response = llm_client.chat.completions.create(
                    model=chat_model,
                    messages=llm_messages,
                    temperature=0.3,
                )

                st.session_state["system_prompt_sent"] = True
                assistant_text = response.choices[0].message.content

        except Exception as e:
            assistant_text = f"An error occurred while generating the answer: {e}"
            sources_data = []
            st.error(assistant_text)

    st.session_state["chat_messages"].append({"role": "assistant", "content": assistant_text})
    st.session_state["sources_by_turn"].append(sources_data)
    save_session_data(st.session_state["session_id"])
    st.rerun()


st.markdown("---")
st.subheader("Session log")

log_col1, log_col2, log_col3 = st.columns([1, 1, 0.5])

with log_col1:
    if st.button("Append new messages to log"):
        current_msg_count = len(st.session_state["chat_messages"])
        current_source_count = len(st.session_state["sources_by_turn"])

        last_msg_count = st.session_state.get("log_msg_count", 0)
        last_source_count = st.session_state.get("log_source_count", 0)

        if current_msg_count > last_msg_count or current_source_count > last_source_count:
            new_text = format_log_entries(
                st.session_state["chat_messages"],
                st.session_state["sources_by_turn"],
                start_msg_index=last_msg_count,
                start_source_index=last_source_count,
            )

            current_log = st.session_state.get("editable_log", "")
            if current_log:
                st.session_state["editable_log"] = current_log.strip() + "\n\n" + new_text
            else:
                st.session_state["editable_log"] = new_text

            st.session_state["log_msg_count"] = current_msg_count
            st.session_state["log_source_count"] = current_source_count

            save_session_data(st.session_state["session_id"])
            st.success("The log was updated with new messages.")
            st.rerun()
        else:
            st.info("There are no new messages to add.")

with log_col3:
    if st.button("Clear session"):
        st.session_state["chat_messages"] = []
        st.session_state["sources_by_turn"] = []
        st.session_state["editable_log"] = ""
        st.session_state["log_msg_count"] = 0
        st.session_state["log_source_count"] = 0
        st.session_state["system_prompt_sent"] = False
        save_session_data(st.session_state["session_id"])
        st.rerun()

edited_log = st.text_area(
    "Editable log",
    value=st.session_state.get("editable_log", ""),
    height=320,
)

if edited_log != st.session_state.get("editable_log", ""):
    st.session_state["editable_log"] = edited_log
    save_session_data(st.session_state["session_id"])

dl_col1, dl_col2, dl_col3 = st.columns(3)

with dl_col1:
    st.download_button(
        label="Download edited log (.md)",
        data=st.session_state.get("editable_log", ""),
        file_name=f"interview_edited_log_{st.session_state['session_id']}.md",
        mime="text/markdown",
    )

with dl_col2:
    pdf_bytes = generate_log_pdf(st.session_state.get("editable_log", ""))
    st.download_button(
        label="Download edited log (.pdf)",
        data=pdf_bytes,
        file_name=f"interview_session_{st.session_state['session_id']}.pdf",
        mime="application/pdf",
    )

with dl_col3:
    full_log_content = generate_full_log_text(
        st.session_state["chat_messages"],
        st.session_state["sources_by_turn"],
    )
    st.download_button(
        label="Download full raw log (.md)",
        data=full_log_content,
        file_name=f"interview_full_log_{st.session_state['session_id']}.md",
        mime="text/markdown",
    )


@st.dialog("Save session")
def save_session_dialog():
    st.write("Name this checkpoint. The session ID is stored automatically.")
    session_name = st.text_input("Checkpoint name", key="save_session_name")
    save_col, cancel_col = st.columns(2)

    with save_col:
        if st.button("Save", type="primary", key="confirm_save_session"):
            if not session_name.strip():
                st.warning("Checkpoint name is required.")
                return

            saved_path = save_session_checkpoint(
                session_id=st.session_state["session_id"],
                session_name=session_name,
            )
            st.session_state["last_saved_checkpoint"] = str(saved_path)
            st.rerun()

    with cancel_col:
        if st.button("Cancel", key="cancel_save_session"):
            st.rerun()


if st.button("Save session checkpoint"):
    save_session_dialog()

if st.session_state.get("last_saved_checkpoint"):
    st.success(f"Session checkpoint saved: `{Path(st.session_state['last_saved_checkpoint']).name}`")
    del st.session_state["last_saved_checkpoint"]
