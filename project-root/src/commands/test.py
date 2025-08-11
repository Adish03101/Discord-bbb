# my code
import discord
import sys
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import json, hashlib
from discord.commands import Option
import os
import logging
import json, hashlib, pickle
from pathlib import Path
import faiss
import requests
from rank_bm25 import BM25Okapi
sys.path.append("/home/black/Backup/Discord-bbb/project-root/src")
from llm_actions.test_rag import (
    rag_generate,
    embeddings_from_texts,
    bm25_sparse_from_texts,
    faiss_index_from_texts,
)
from core.pg_db import get_pool

def _sha1(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8", "ignore")).hexdigest()

def _file_sig(p: Path) -> str:
    try:
        st = p.stat()
        return f"{p.name}|{st.st_mtime_ns}|{st.st_size}"
    except FileNotFoundError:
        return f"{p.name}|0|0"

def list_note_files(base_dir: Path) -> list[Path]:
    """List all .md and .txt files in the given directory recursively."""
    files: list[Path] = []
    if not base_dir.exists():
        return files
    for p in base_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".md", ".txt"}:
            files.append(p)
    return files

def ensure_index_bundle_from_files(
    files: list[Path],
    index_dir: Path,
) -> tuple[list[str], "BM25Okapi", faiss.Index]:
    """
    Persist bundle in `index_dir`:
      faiss.index, bm25.pkl, chunks.pkl, manifest.json

    Embeds only files whose signature wasn't seen before.
    Returns: (chunks, bm25, faiss_index)
    """
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss_path    = index_dir / "faiss.index"
    manifest_path = index_dir / "manifest.json"
    chunks_path   = index_dir / "chunks.pkl"
    bm25_path     = index_dir / "bm25.pkl"

    # Load prior state
    try:
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"hashes": [], "files": {}}
    except Exception:
        manifest = {"hashes": [], "files": {}}

    if chunks_path.exists():
        try:
            chunks: list[str] = pickle.loads(chunks_path.read_bytes())
        except Exception:
            chunks = []
    else:
        chunks = []

    # Decide which files are new/changed
    new_files: list[Path] = []
    for p in files:
        sig = _file_sig(p)
        rel = str(p)  # or vault-relative path if you prefer
        prev = manifest["files"].get(rel)
        if prev is None or prev.get("sig") != sig:
            new_files.append(p)

    # If nothing exists yet, force first-run embed of all files
    if not faiss_path.exists() and chunks == []:
        new_files = files[:]

    # Read contents only for new/changed files
    new_texts: list[str] = []
    new_hashes: list[str] = []
    for p in new_files:
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        new_texts.append(text)
        new_hashes.append(_sha1(text))
        manifest["files"][str(p)] = {"sig": _file_sig(p)}

    # Update FAISS via your RAG helper (append only)
    if new_texts:
        # uses your faiss_index_from_texts(...) which loads-existing + adds + writes back
        index = faiss_index_from_texts(new_texts, index_path=index_dir, index_name="faiss")
        chunks = chunks + new_texts
        manifest["hashes"].extend(new_hashes)
        manifest_path.write_text(json.dumps(manifest, indent=2))
        chunks_path.write_bytes(pickle.dumps(chunks))
        # Rebuild BM25 to match chunks (fast) and persist
        bm25 = bm25_sparse_from_texts(chunks)
        bm25_path.write_bytes(pickle.dumps(bm25))
    else:
        # Nothing new: load FAISS + BM25/chunks
        if faiss_path.exists():
            index = faiss.read_index(str(faiss_path))
        else:
            # Safety: if we have files but no index for some reason, build once
            texts_all = []
            for p in files:
                try:
                    texts_all.append(p.read_text(encoding="utf-8", errors="ignore"))
                except Exception:
                    pass
            index = faiss_index_from_texts(texts_all, index_path=index_dir, index_name="faiss")
            chunks = texts_all
            manifest["hashes"] = [_sha1(t) for t in texts_all]
            for p in files:
                manifest["files"][str(p)] = {"sig": _file_sig(p)}
            manifest_path.write_text(json.dumps(manifest, indent=2))
            chunks_path.write_bytes(pickle.dumps(chunks))
        if bm25_path.exists():
            try:
                bm25 = pickle.loads(bm25_path.read_bytes())
            except Exception:
                bm25 = bm25_sparse_from_texts(chunks)
                bm25_path.write_bytes(pickle.dumps(bm25))
        else:
            bm25 = bm25_sparse_from_texts(chunks)
            bm25_path.write_bytes(pickle.dumps(bm25))

    return chunks, bm25, index

# ——— CONFIGURATION —————————————————————————————————————————————
load_dotenv()  # FIX: call the function
DATABASE_URL = os.getenv("POSTGRES_URL")
RAG_INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", "./indices"))
DATA_DIR = Path(os.getenv("STORAGE_DIR", "./data/files"))
TOKEN = os.getenv("PROJECT_TOKEN")

# Add HF configuration for LLM calls
HF_GEN_MODEL = os.getenv("HF_GEN_MODEL", "openai/gpt-oss-20b:fireworks-ai")
HF_API_URL = "https://router.huggingface.co/v1/chat/completions"
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("task_bot")

bot = discord.Bot(intents=discord.Intents.default())

# FIX: make this async and await the async fetcher
async def get_channel_path(discord_channel_id: str) -> Path | None:
    """Build /vault/anc0/anc1/.../name for a Discord channel row."""
    pool = await get_pool()
    if not pool:
        logger.error("Failed to connect to the database.")
        return None

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, name, ancestors
            FROM public.channel
            WHERE discord_channel_id = $1
            """,
            str(discord_channel_id),
        )
        if not row:
            logger.warning(f"No channel found for discord_channel_id={discord_channel_id}")
            return None

        # ancestors are highest-first as per your schema
        pieces = list(row["ancestors"] or []) + [row["name"]]

        # FIX: Define VAULT_DIR - this was missing!
        os.environ["OBSIDIAN_VAULT_ROOT"] = "/home/black/First"
        VAULT_DIR = Path(os.getenv("OBSIDIAN_VAULT_ROOT", "./vault"))
        path_to_fetch = VAULT_DIR
        for piece in pieces:
            path_to_fetch /= piece

        return path_to_fetch

def hf_llm_call(prompt: str) -> str:
    """Call Hugging Face LLM API."""
    payload = {
        "model": HF_GEN_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful and concise assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
    }
    try:
        resp = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return "Sorry, I couldn't generate a response at this time."

@bot.event
async def on_ready():
    """Initialize bot with proper error handling"""
    try:
        logger.info("Task Bot logged in as %s", bot.user)
        print(f"Task Bot logged in as {bot.user}")
    except Exception as e:
        logger.error(f"Failed to initialize bot: {e}")
        raise

@bot.slash_command(name="project_query", description="Ask a question about this project")
async def project_query(ctx: discord.ApplicationContext,
                        query: Option(str, "Your question")):
    """Handle project queries using RAG."""
    try:
        await ctx.defer()
    except discord.errors.NotFound:
        logger.warning("Interaction already expired; skipping defer.")

    # Get the base directory for this channel
    base_dir = await get_channel_path(str(ctx.channel_id))
    if not base_dir:
        await ctx.respond("No project folder found for this channel.")
        return

    # List all note files in the directory
    files = list_note_files(base_dir)
    if not files:
        await ctx.respond("No relevant documents found in this channel.")
        return

    # Create or load the index bundle from files
    try:
        chunks, bm25, faiss_index = ensure_index_bundle_from_files(files, base_dir / "_index")
    except Exception as e:
        logger.error(f"Failed to create index bundle: {e}")
        await ctx.respond("Failed to process documents for this channel.")
        return

    # Generate answer using RAG
    try:
        answer = rag_generate(
            query=query,
            chunks=chunks,
            bm25=bm25,
            faiss_index=faiss_index,
            embeddings_from_texts=embeddings_from_texts,
            llm_call=hf_llm_call,
            top_k=5,
            alpha=0.3,
        )
        await ctx.respond(answer or "No answer could be generated.")
    except Exception as e:
        logger.error(f"RAG generation failed: {e}")
        await ctx.respond("Sorry, I encountered an error while generating the answer.")

bot.run(TOKEN)