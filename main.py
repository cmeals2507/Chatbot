"""
Streamlit-powered Chatbot with optional document grounding.

This application provides a simple conversational interface backed by
OpenAI's Chat Completion API.  It exposes the ``gpt‑4.1‑mini`` model with a
moderate temperature (0.5) and maintains a running conversation using
``st.session_state``.  Users can optionally upload one or more documents to
"ground" the chat; the contents of each file are summarised using the same
model, and those summaries are passed to the assistant on each turn.  A
list of the uploaded files appears in the sidebar with tool‑tips showing
their respective summaries.

Before running this app you must set the ``OPENAI_API_KEY`` environment
variable.  You will also need to install the dependencies listed in
``requirements.txt`` within a virtual environment.  See the README or
deployment notes for further details.
"""


FIXED_TEMP_PREFIXES = ("o1", "o3", "o4")

def _supports_temperature(model: str) -> bool:
    m = (model or "").lower()
    return not any(m.startswith(p) for p in FIXED_TEMP_PREFIXES)

import os
from io import BytesIO
from typing import Dict, List
import json
import pprint
import pathlib
# --- RAG cache locations (v2) ---
from pathlib import Path as _PathAlias  # avoid confusion with existing pathlib
CACHE_DIR = pathlib.Path(__file__).parent / ".cache" / "rag"
GROUNDING_CACHE = CACHE_DIR / "grounding_cache.v2.json"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
import PyPDF2
import tiktoken
import concurrent.futures
import hashlib
import time
from streamlit.runtime.scriptrunner import add_script_run_ctx
def token_truncate(text: str, max_tokens: int, model: str = "gpt-4.1-mini") -> str:
    """Truncate text to the first max_tokens tokens using tiktoken."""
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    truncated = tokens[:max_tokens]
    return enc.decode(truncated)


def file_hash(path: str) -> str:
    """Compute SHA256 hash of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


import numpy as np
import streamlit as st
from openai import OpenAI

# Upload constraints (grounding has no cap)
UPLOAD_MAX_FILES = 5
UPLOAD_MAX_MB = 20
SUPPORTED_GROUNDING_EXTS = {".json", ".pdf"}

# Configure the API key from the environment.  This will raise if the key
# isn't set, making it obvious what went wrong.
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError(
        "The OPENAI_API_KEY environment variable is not set. "
        "Please export your API key before launching the app."
    )
client = OpenAI()


def summarise_document(text: str, *, model: str = "o4-mini", temperature: float = 0.5) -> str:
    """Generate a short summary of ``text`` using the ChatCompletion API.

    The text is truncated to a manageable length before being sent to
    the model to avoid token limit issues.  If the request fails the
    exception is captured and returned to the user instead of
    crashing the app.

    Parameters
    ----------
    text:
        The full contents of the document to be summarised.
    model:
        Name of the OpenAI model to use for summarisation.
    temperature:
        Sampling temperature for the summary.  Lower values make
        responses more deterministic.

    Returns
    -------
    str
        A summary of the provided text, or an error message.
    """
    # Guard against empty documents
    if not text.strip():
        return "(empty document)"

    # Limit the amount of text sent to the model.  GPT‑4.1‑mini can handle
    # around 8k tokens, but we truncate conservatively to avoid
    # over‑length requests.  Each character roughly equates to one token
    # after encoding, so this limit should be safe.
    max_chars = 4000
    snippet = text[:max_chars]

    # Attempt to detect JSON and pretty-print it to avoid JSON formatting issues
    try:
        stripped = snippet.lstrip()
        if stripped.startswith("{") or stripped.startswith("["):
            parsed_json = json.loads(snippet)
            snippet = pprint.pformat(parsed_json, width=80)
    except Exception:
        # If JSON parsing fails, leave snippet as is
        pass

    # Ensure snippet is a safe string (handle any binary or structured data gracefully)
    if not isinstance(snippet, str):
        snippet = str(snippet)

    try:
        kwargs = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an assistant summarising documents for a user. "
                        "Produce a concise paragraph capturing the key ideas of "
                        "the following text."
                    ),
                },
                {"role": "user", "content": snippet},
            ]
        }
        if _supports_temperature(model):
            kwargs["temperature"] = temperature
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content.strip()
    except Exception as exc:  # noqa: BLE001
        print(f"Summarisation error: {exc}")
        return f"Summarisation failed: {exc}"


def build_context_from_summaries(file_data: Dict[str, Dict[str, str]]) -> str:
    """Construct a context string from the stored summaries for the chat API.

    If a file has no summary or the summary starts with 'Summarisation failed',
    fall back to a preview of the raw file content (first 2000 characters).
    Prefix raw content with '(raw preview)'.
    """
    parts: List[str] = []
    for name, data in file_data.items():
        summary = data.get("summary", "")
        # Fallback to raw content if no summary or summary failed
        if (
            not summary
            or summary.strip().lower().startswith("summarisation failed")
        ):
            raw_content = data.get("content", "")
            preview = raw_content[:2000]
            parts.append(
                f"File: {name}\nSummary: (raw preview)\n{preview}"
            )
        else:
            parts.append(f"File: {name}\nSummary: {summary}")
    return "\n\n".join(parts)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D numpy arrays."""
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def embed_text(text: str) -> np.ndarray:
    """Embed text using the OpenAI embeddings API and return a numpy array."""
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small",
        )
        embedding = response.data[0].embedding
        return np.array(embedding, dtype=np.float32)
    except Exception:
        return np.zeros(1536, dtype=np.float32)  # fallback zero vector with typical embedding size

# Batch embedding helper
from typing import List
def embed_texts(texts: List[str], batch_size: int = 64) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            resp = client.embeddings.create(input=batch, model="text-embedding-3-small")
            for d in resp.data:
                out.append(np.array(d.embedding, dtype=np.float32))
        except Exception:
            # Fallback per item if batch fails
            for t in batch:
                out.append(embed_text(t))
    return out


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into chunks of approximately chunk_size with overlap."""
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks



def load_grounding_files(progress_callback=None) -> None:
    """
    Load grounding files from the 'grounding' subdirectory, summarise and embed them.
    Uses a cache file (.grounding_cache.json) to avoid recomputation for unchanged files.
    Processes files in parallel for speed.
    """
    grounding_path = pathlib.Path(__file__).parent / "grounding"
    if not grounding_path.exists() or not grounding_path.is_dir():
        return

    cache_path = GROUNDING_CACHE
    cache = {}
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    if "file_data" not in st.session_state:
        st.session_state["file_data"] = {}
    if "embeddings" not in st.session_state:
        st.session_state["embeddings"] = []

    files_to_process = []
    file_paths = []
    for file_path in grounding_path.glob("*"):
        if file_path.suffix.lower() not in {".json", ".pdf"}:
            continue
        name = file_path.name
        hash_val = file_hash(str(file_path))
        cache_key = f"{name}:{hash_val}"
        if (
            name in st.session_state["file_data"]
            and any(e.get("file") == name for e in st.session_state["embeddings"])
        ):
            # Already loaded in session
            continue
        if cache_key in cache:
            # Load from cache (backward compatible, handle both "embeddings" and "embeddings_cache")
            st.session_state["file_data"][name] = cache[cache_key].get("file_data", {})
            emb_list = cache[cache_key].get("embeddings_cache", cache[cache_key].get("embeddings", []))
            for emb in emb_list:
                emb_session = dict(emb)
                if isinstance(emb_session.get("embedding"), list):
                    emb_session["embedding"] = np.array(emb_session["embedding"], dtype=np.float32)
                st.session_state["embeddings"].append(emb_session)
            if progress_callback:
                progress_callback(f"Loaded {name} from cache")
            continue
        files_to_process.append((file_path, name, hash_val, cache_key))
        file_paths.append(file_path)

    def process_file(args):
        file_path, name, hash_val, cache_key = args
        text = ""
        text_for_summary_source = ""
        text_for_embedding_source = ""
        try:
            if file_path.suffix.lower() == ".json":
                with open(file_path, "r", encoding="utf-8") as f:
                    parsed_json = json.load(f)
                pretty_text = pprint.pformat(parsed_json, width=80)
                compact_text = json.dumps(parsed_json, separators=(",", ":"), ensure_ascii=False)
                text = pretty_text
                text_for_summary_source = compact_text
                text_for_embedding_source = compact_text
            elif file_path.suffix.lower() == ".pdf":
                with open(file_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    texts = []
                    for page in reader.pages:
                        try:
                            page_text = page.extract_text()
                            if page_text:
                                texts.append(page_text)
                        except Exception:
                            pass
                    text = "\n\n".join(texts)
                text_for_summary_source = text
                text_for_embedding_source = text
        except Exception:
            return None

        # Truncate for summary (safe limit: 3000 tokens)
        text_for_summary = token_truncate(text_for_summary_source, 3000)
        # Summary skipping logic
        generate_summaries = st.session_state.get("generate_summaries", False)
        if generate_summaries:
            summary = summarise_document(text_for_summary, model=st.session_state.get("model", "o4-mini"))
        else:
            summary = ""
        # Truncate for embedding (safe chunk: 500 tokens per chunk)
        chunks = []
        for raw_chunk in chunk_text(text_for_embedding_source, chunk_size=2000, overlap=200):
            chunk_txt = token_truncate(raw_chunk, 500)
            chunks.append(chunk_txt)
        # Batch embeddings
        embeddings_np = embed_texts(chunks)
        # Build two parallel lists: one for cache (list), one for session (np.array)
        embeddings_for_cache = []
        embeddings_for_session = []
        for chunk, emb_np in zip(chunks, embeddings_np):
            emb_cache = {
                "embedding": emb_np.tolist(),
                "text": chunk,
                "file": name,
            }
            emb_session = {
                "embedding": emb_np,
                "text": chunk,
                "file": name,
            }
            embeddings_for_cache.append(emb_cache)
            embeddings_for_session.append(emb_session)
        return {
            "name": name,
            "hash_val": hash_val,
            "cache_key": cache_key,
            "file_data": {"content": text, "summary": summary},
            "embeddings_cache": embeddings_for_cache,
            "embeddings_session": embeddings_for_session,
        }

    results = []
    if files_to_process:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, (os.cpu_count() or 4) * 2)) as executor:
            future_to_file = {}
            for args in files_to_process:
                fut = executor.submit(process_file, args)
                add_script_run_ctx(fut)
                future_to_file[fut] = args[1]
            for future in concurrent.futures.as_completed(future_to_file):
                name = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        # Update session state immediately for responsiveness
                        st.session_state["file_data"][result["name"]] = result["file_data"]
                        for emb in result["embeddings_session"]:
                            st.session_state["embeddings"].append(emb)
                        # Update cache
                        cache[result["cache_key"]] = {
                            "file_data": result["file_data"],
                            "embeddings_cache": result["embeddings_cache"],
                        }
                        if progress_callback:
                            progress_callback(f"Loaded {result['name']} (processed)")
                except Exception as exc:
                    if progress_callback:
                        progress_callback(f"Failed loading {name}: {exc}")
    # Save cache (ensure directory exists)
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except Exception:
        pass



def main() -> None:
    """Run the Streamlit chat application."""
    st.set_page_config(page_title="Chatbot with RAG", layout="wide")
    st.title("Chatbot with Optional Grounding")

    # Initialise session state containers if this is the first run
    if "messages" not in st.session_state:
        st.session_state["messages"]: List[Dict[str, str]] = []
    if "file_data" not in st.session_state:
        st.session_state["file_data"]: Dict[str, Dict[str, str]] = {}
    if "embeddings" not in st.session_state:
        st.session_state["embeddings"]: List[Dict[str, object]] = []
    if "model" not in st.session_state:
        st.session_state["model"] = "o4-mini"
    if "k" not in st.session_state:
        st.session_state["k"] = 4

    # Render layout first, then load files (so UI appears immediately)
    sidebar_placeholder = st.sidebar.empty()
    main_placeholder = st.empty()

    def sidebar_content():
        with sidebar_placeholder.container():
            st.header("Settings")
            model_options = ['gpt-4.1', 'gpt-4.1-mini', 'o4-mini', 'o3-mini', 'gpt-4o', 'gpt-4.5-preview']
            selected_model = st.selectbox("Select model", model_options, index=model_options.index(st.session_state["model"]))
            st.session_state["model"] = selected_model

            k = st.slider("Number of retrieved chunks (k)", min_value=1, max_value=10, value=st.session_state["k"])
            st.session_state["k"] = k

            # Add checkbox to skip/generate summaries
            generate_summaries = st.checkbox("Generate summaries (for hover)", value=True)
            st.session_state["generate_summaries"] = generate_summaries

            st.header("Document Grounding")
            # Clarify that only uploads are capped; grounding folder is unlimited
            grounding_path = pathlib.Path(__file__).parent / "grounding"
            try:
                grounding_count = sum(
                    1 for p in grounding_path.glob("*") if p.suffix.lower() in SUPPORTED_GROUNDING_EXTS
                ) if grounding_path.exists() else 0
            except Exception:
                grounding_count = 0
            st.caption(
                f"Grounding folder: unlimited files (currently detected: {grounding_count}).\n"
                f"Uploads: up to {UPLOAD_MAX_FILES} files, ≤ {UPLOAD_MAX_MB} MB each."
            )

            uploaded_files = st.file_uploader(
                f"Upload (max {UPLOAD_MAX_FILES} files, ≤ {UPLOAD_MAX_MB} MB each)",
                accept_multiple_files=True,
                type=None,
            )
            # Enforce max files manually to avoid Streamlit version differences
            if uploaded_files and len(uploaded_files) > UPLOAD_MAX_FILES:
                st.warning(f"Only the first {UPLOAD_MAX_FILES} files will be processed.")
                uploaded_files = uploaded_files[:UPLOAD_MAX_FILES]
            return uploaded_files

    uploaded_files = sidebar_content()

    # Show progress/loading spinner for grounding files
    progress_placeholder = st.empty()
    if "grounding_loaded" not in st.session_state:
        progress_msgs = []
        def progress_callback(msg):
            progress_msgs.append(msg)
            progress_placeholder.info("\n".join(progress_msgs))
        with st.spinner("Loading grounding files..."):
            load_grounding_files(progress_callback=progress_callback)
        st.session_state["grounding_loaded"] = True
        progress_placeholder.empty()
    else:
        progress_placeholder.empty()

    # Upload cache path for caching uploaded files
    UPLOAD_CACHE_PATH = pathlib.Path(__file__).parent / ".upload_cache.json"
    upload_cache = {}
    if UPLOAD_CACHE_PATH.exists():
        try:
            upload_cache = json.load(open(UPLOAD_CACHE_PATH, "r", encoding="utf-8"))
        except Exception:
            upload_cache = {}

    # Handle uploaded files with truncation and parallel processing, with caching
    if uploaded_files:
        # Only process new files
        new_files = []
        for uploaded_file in uploaded_files:
            if uploaded_file.size > UPLOAD_MAX_MB * 1024 * 1024:
                st.warning(f"File {uploaded_file.name} skipped: exceeds {UPLOAD_MAX_MB}MB size limit.")
                continue
            name = uploaded_file.name
            raw_bytes = uploaded_file.read()
            u_hash = hashlib.sha256(raw_bytes).hexdigest()
            u_key = f"{name}:{u_hash}"
            if u_key in upload_cache:
                # Load from cache: populate file_data and embeddings in session, skip reprocessing
                cached = upload_cache[u_key]
                st.session_state["file_data"][name] = cached["file_data"]
                for emb in cached.get("embeddings_cache", []):
                    emb_session = dict(emb)
                    emb_session["embedding"] = np.array(emb["embedding"], dtype=np.float32)
                    st.session_state["embeddings"].append(emb_session)
                continue
            if name not in st.session_state["file_data"]:
                new_files.append((name, raw_bytes, u_key))
        def process_uploaded_file(args):
            name, raw_bytes, u_key = args
            # Try to decode as UTF-8; fallback to str
            try:
                text = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                text = str(raw_bytes)
            # Detect JSON by leading { or [
            text_stripped = text.lstrip()
            if text_stripped.startswith("{") or text_stripped.startswith("["):
                try:
                    parsed_json = json.loads(text)
                    pretty_text = pprint.pformat(parsed_json, width=80)
                    compact_text = json.dumps(parsed_json, separators=(",", ":"), ensure_ascii=False)
                    text_display = pretty_text
                    text_for_summary_source = compact_text
                    text_for_embedding_source = compact_text
                except Exception:
                    text_display = text
                    text_for_summary_source = text
                    text_for_embedding_source = text
            else:
                text_display = text
                text_for_summary_source = text
                text_for_embedding_source = text
            # Truncate for summary (safe limit: 3000 tokens)
            text_for_summary = token_truncate(text_for_summary_source, 3000)
            generate_summaries = st.session_state.get("generate_summaries", False)
            if generate_summaries:
                summary = summarise_document(text_for_summary, model=st.session_state["model"])
            else:
                summary = ""
            # Truncate for embedding (safe chunk: 500 tokens per chunk)
            chunks = []
            for raw_chunk in chunk_text(text_for_embedding_source, chunk_size=2000, overlap=200):
                chunk_txt = token_truncate(raw_chunk, 500)
                chunks.append(chunk_txt)
            embeddings_np = embed_texts(chunks)
            embeddings_session = []
            embeddings_cache = []
            for chunk, emb_np in zip(chunks, embeddings_np):
                embeddings_session.append({
                    "embedding": emb_np,
                    "text": chunk,
                    "file": name,
                })
                embeddings_cache.append({
                    "embedding": emb_np.tolist(),
                    "text": chunk,
                    "file": name,
                })
            return {
                "name": name,
                "file_data": {"content": text_display, "summary": summary},
                "embeddings_session": embeddings_session,
                "embeddings_cache": embeddings_cache,
                "u_key": u_key,
            }
        if new_files:
            upload_progress = st.sidebar.empty()
            upload_msgs = []
            def upload_progress_callback(msg):
                upload_msgs.append(msg)
                upload_progress.info("\n".join(upload_msgs))
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, (os.cpu_count() or 4) * 2)) as executor:
                future_to_name = {}
                for args in new_files:
                    fut = executor.submit(process_uploaded_file, args)
                    add_script_run_ctx(fut)
                    future_to_name[fut] = args[0]
                for future in concurrent.futures.as_completed(future_to_name):
                    name = future_to_name[future]
                    try:
                        result = future.result()
                        if result:
                            st.session_state["file_data"][result["name"]] = result["file_data"]
                            for emb in result["embeddings_session"]:
                                st.session_state["embeddings"].append(emb)
                            # Store in upload_cache for future runs
                            upload_cache[result["u_key"]] = {
                                "file_data": result["file_data"],
                                "embeddings_cache": result["embeddings_cache"],
                            }
                            upload_progress_callback(f"Uploaded {result['name']} loaded.")
                    except Exception as exc:
                        upload_progress_callback(f"Failed loading {name}: {exc}")
            upload_progress.empty()
        # Save updated upload_cache back to disk after processing uploads
        try:
            with open(UPLOAD_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(upload_cache, f)
        except Exception:
            pass

    # Display uploaded files with summaries in an expander and tool-tips for quick hover
    if st.session_state["file_data"]:
        with st.sidebar:
            st.markdown("**Grounding & Uploaded Files**")
            with st.expander("View summaries", expanded=False):
                for name, data in st.session_state["file_data"].items():
                    summary = data.get("summary") or "(no summary)"
                    preview = (summary[:180] + "…") if len(summary) > 200 else summary
                    st.markdown(f"**{name}**\n\n{preview}")
            # Keep quick hover as well for users who like tooltips
            for name, data in st.session_state["file_data"].items():
                st.checkbox(name, key=f"file_{name}", help=data.get("summary") or "No summary available")

    # Display the conversation so far
    for msg in st.session_state["messages"]:
        role = msg.get("role")
        content = msg.get("content")
        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)
        elif role == "assistant":
            with st.chat_message("assistant"):
                st.markdown(content)

    # Paperclip popover for per-turn attachments + paired text
    popover_cols = st.columns([1, 9])
    with popover_cols[0]:
        with st.popover("📎 Attach"):
            st.markdown("**Attach files for this message**")
            st.session_state["attach_files"] = st.file_uploader(
                "Files (JSON/PDF/TXT/MD)",
                accept_multiple_files=True,
                type=["json", "pdf", "txt", "md"],
                key="attach_files_uploader",
            )
            st.session_state["attach_note"] = st.text_area(
                "Notes/context for these files (optional)",
                key="attach_note_input",
                height=100,
                placeholder="e.g., Focus on Section 3 and the 'methods' field"
            )
    # Small status next to the input showing pending attachments
    with popover_cols[1]:
        names = []
        if st.session_state.get("attach_files"):
            names = [f.name for f in st.session_state["attach_files"]]
        note_preview = st.session_state.get("attach_note") or ""
        if names or note_preview:
            preview_note = (note_preview[:60] + "…") if len(note_preview) > 60 else note_preview
            st.caption("📎 Pending: " + (", ".join(names) if names else "no files") + (f" — {preview_note}" if preview_note else ""))

    # User input for the next turn
    user_input = st.chat_input("Your message")
    if user_input:
        # Gather pending attachments from the popover
        attached_files = st.session_state.get("attach_files") or []
        attach_note = st.session_state.get("attach_note") or ""

        # Show the user's message immediately so it appears in this run
        with st.chat_message("user"):
            st.markdown(user_input)
            if attached_files:
                st.caption("📎 Files: " + ", ".join(f.name for f in attached_files))
            if attach_note:
                st.caption("📝 Note: " + attach_note)

        # Append the user input to the message history
        st.session_state["messages"].append({"role": "user", "content": user_input})

        # If there are turn attachments, process them now (summaries + embeddings)
        if attached_files:
            new_files = []
            for uploaded_file in attached_files:
                if uploaded_file.size > UPLOAD_MAX_MB * 1024 * 1024:
                    st.warning(f"File {uploaded_file.name} skipped: exceeds {UPLOAD_MAX_MB}MB size limit.")
                    continue
                name = uploaded_file.name
                raw_bytes = uploaded_file.read()
                u_hash = hashlib.sha256(raw_bytes).hexdigest()
                u_key = f"{name}:{u_hash}"

                # Try upload cache first
                cached = None
                if os.path.exists(str(pathlib.Path(__file__).parent / ".upload_cache.json")):
                    try:
                        _uc = json.load(open(pathlib.Path(__file__).parent / ".upload_cache.json", "r", encoding="utf-8"))
                        cached = _uc.get(u_key)
                    except Exception:
                        cached = None
                if cached:
                    st.session_state["file_data"][name] = cached["file_data"]
                    for emb in cached.get("embeddings_cache", []):
                        emb_session = dict(emb)
                        emb_session["embedding"] = np.array(emb["embedding"], dtype=np.float32)
                        st.session_state["embeddings"].append(emb_session)
                    continue

                # Decode text
                try:
                    text = raw_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    text = str(raw_bytes)

                # Detect JSON
                text_stripped = text.lstrip()
                if text_stripped.startswith("{") or text_stripped.startswith("["):
                    try:
                        parsed_json = json.loads(text)
                        pretty_text = pprint.pformat(parsed_json, width=80)
                        compact_text = json.dumps(parsed_json, separators=(",", ":"), ensure_ascii=False)
                        text_display = pretty_text
                        text_for_summary_source = compact_text
                        text_for_embedding_source = compact_text
                    except Exception:
                        text_display = text
                        text_for_summary_source = text
                        text_for_embedding_source = text
                else:
                    text_display = text
                    text_for_summary_source = text
                    text_for_embedding_source = text

                # Summarise (respect toggle) and embed
                text_for_summary = token_truncate(text_for_summary_source, 3000)
                if st.session_state.get("generate_summaries", True):
                    summary = summarise_document(text_for_summary, model=st.session_state["model"])
                else:
                    summary = ""

                chunks = []
                for raw_chunk in chunk_text(text_for_embedding_source, chunk_size=2000, overlap=200):
                    chunks.append(token_truncate(raw_chunk, 500))
                vectors = embed_texts(chunks)
                embeddings_session = []
                embeddings_cache = []
                for chunk, vec in zip(chunks, vectors):
                    embeddings_session.append({"embedding": vec, "text": chunk, "file": name})
                    embeddings_cache.append({"embedding": vec.tolist(), "text": chunk, "file": name})

                # Update state and cache
                st.session_state["file_data"][name] = {"content": text_display, "summary": summary}
                st.session_state["embeddings"].extend(embeddings_session)

                up_path = pathlib.Path(__file__).parent / ".upload_cache.json"
                try:
                    _uc = {}
                    if up_path.exists():
                        _uc = json.load(open(up_path, "r", encoding="utf-8"))
                    _uc[u_key] = {"file_data": {"content": text_display, "summary": summary}, "embeddings_cache": embeddings_cache}
                    json.dump(_uc, open(up_path, "w", encoding="utf-8"))
                except Exception:
                    pass

        # Clear pending attachments after sending this turn
        st.session_state["attach_files"] = []
        st.session_state["attach_note"] = ""

        # Build the context string from file summaries
        context = build_context_from_summaries(st.session_state["file_data"])

        # Compose the message list for the API call
        api_messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant conversing with a user. "
                    "Use the provided document summaries to ground your answers when relevant."
                ),
            },
        ]
        if context:
            api_messages.append(
                {
                    "role": "system",
                    "content": (
                        "Here are some documents uploaded by the user. "
                        "Refer to them if they are relevant to the question:\n\n"
                        + context
                    ),
                }
            )

        # If user provided an attachment note, include it as grounding hint
        if attach_note:
            api_messages.append({
                "role": "system",
                "content": "Attachment note from the user (use as guidance if relevant):\n" + attach_note,
            })

        # Vector search: embed query, find top k similar chunks and append as system message
        if st.session_state["embeddings"]:
            try:
                query_embedding = embed_text(user_input)
                similarities = []
                for item in st.session_state["embeddings"]:
                    sim = cosine_similarity(query_embedding, item["embedding"])
                    similarities.append((sim, item))
                similarities.sort(key=lambda x: x[0], reverse=True)
                top_k = similarities[: st.session_state["k"]]
                retrieved_chunks = [f"File: {item['file']}\nContent: {item['text']}" for _, item in top_k if _ > 0]
                if retrieved_chunks:
                    api_messages.append(
                        {
                            "role": "system",
                            "content": (
                                "The following text chunks are most relevant to the user's query:\n\n"
                                + "\n\n---\n\n".join(retrieved_chunks)
                            ),
                        }
                    )
            except Exception:
                # Fail silently on embedding or similarity errors
                pass

        # Append the conversation history
        api_messages += st.session_state["messages"]

        # Call the OpenAI API for the assistant's reply
        try:
            kwargs = {
                "model": st.session_state["model"],
                "messages": api_messages,
            }
            if _supports_temperature(st.session_state["model"]):
                kwargs["temperature"] = 0.5
            completion = client.chat.completions.create(**kwargs)
            assistant_reply = completion.choices[0].message.content.strip()
        except Exception as exc:  # noqa: BLE001
            assistant_reply = f"Error contacting OpenAI API: {exc}"

        # Append the assistant's reply to the history and display it
        st.session_state["messages"].append({"role": "assistant", "content": assistant_reply})
        with st.chat_message("assistant"):
            st.markdown(assistant_reply)


if __name__ == "__main__":
    main()