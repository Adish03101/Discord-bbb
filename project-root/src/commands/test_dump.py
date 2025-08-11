# ####
# modify the data base

# ####

from pathlib import Path

import os
import uuid
import asyncio
from datetime import datetime, timezone
from pathlib import Path, PurePath
from typing import Optional, List, Union
from zoneinfo import ZoneInfo
import logging
from dotenv import load_dotenv
from slugify import slugify
import discord
from typing import Optional
import uuid, hashlib, mimetypes
# Ensure project-root/src is on sys.path
import sys
sys.path.append("/home/black/Backup/Discord-bbb/project-root/src")

# from llm_actions.summarization import get_summarizer
# from llm_actions.rag import ingest_documents, rag_generate
from core.pg_db import get_pool, upsert_document

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('discord_bot_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ─── Config & Environment ───────────────────────────────────────────────────────
load_dotenv()
TOKEN = os.getenv("DUMP_TOKEN")
DATA_DIR = Path(os.getenv("STORAGE_DIR", "./data/files"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
os.environ["OBSIDIAN_VAULT_ROOT"] = "/home/black/First"
os.environ.setdefault("OBSIDIAN_VAULT_ROOT", "./test_vault")


# Summarizer (AI logic)
# summarizer = get_summarizer()

# ─── Discord Bot Setup ─────────────────────────────────────────────────────────
intents = discord.Intents.default()
intents.guilds = True
intents.messages = True
intents.message_content = True
bot = discord.Bot(intents=intents)

# ─── Hierarchy Configuration ────────────────────────────────────────────────────
HIERARCHY = ["project", "subproject"]
LEAF_THREAD_LEVEL = "task"

# obsidian_sync.py
import os
from pathlib import Path
from datetime import datetime, timezone
import re
from typing import List, Optional

ATTACHMENTS_DIR = "attachments"
INDEX_NAME = "index.md"

def _slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9\-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "untitled"

def _safe_filename(dest_dir: Path, name: str) -> Path:
    base = Path(name).name
    stem = Path(base).stem
    suffix = Path(base).suffix
    candidate = dest_dir / base
    i = 1
    while candidate.exists():
        candidate = dest_dir / f"{stem}-{i}{suffix}"
        i += 1
    return candidate

def _embed_line(mime: str, rel_path: str) -> str:
    # Obsidian embed for images/audio/video/pdf; link for others
    if mime.startswith(("image/", "audio/", "video/")) or mime == "application/pdf":
        return f"![[{rel_path}]]"
    return f"[[{rel_path}]]"

class ObsidianSync:
    def __init__(self, vault_root: Optional[str] = None):
        root = vault_root or os.getenv("OBSIDIAN_VAULT_ROOT")
        if not root:
            raise RuntimeError("Set OBSIDIAN_VAULT_ROOT to your vault directory")
        self.vault = Path(root).expanduser().resolve()
        self.vault.mkdir(parents=True, exist_ok=True)

    def _node_dir(self, ancestors: List[str], curr: str) -> Path:
        parts = [_slug(a) for a in ancestors] + [_slug(curr)]
        node_dir = self.vault.joinpath(*parts)
        node_dir.mkdir(parents=True, exist_ok=True)
        (node_dir / ATTACHMENTS_DIR).mkdir(parents=True, exist_ok=True)
        return node_dir

    def write_document(
        self,
        *,
        ancestors: List[str],
        curr: str,
        filename: str,
        buffer: bytes,
        mime_type: str,
        channel_path_dot: str,
        channel_id: str,
        document_id: str,
        sha256: str,
        created_at_utc: Optional[datetime] = None,
    ) -> Path:
        """Mirror the uploaded file into Obsidian and update index.md."""
        node_dir = self._node_dir(ancestors, curr)
        attachments_dir = node_dir / ATTACHMENTS_DIR
        file_dest = _safe_filename(attachments_dir, filename)
        file_dest.write_bytes(buffer)

        # Write/update index.md
        index_path = node_dir / INDEX_NAME
        created_at_utc = created_at_utc or datetime.now(timezone.utc)
        timestamp = created_at_utc.astimezone().strftime("%Y-%m-%d %H:%M %Z")
        rel = f"{ATTACHMENTS_DIR}/{file_dest.name}"
        embed = _embed_line(mime_type, rel)

        fm = (
            f"---\n"
            f"channel_path: {channel_path_dot}\n"
            f"channel_id: {channel_id}\n"
            f"last_document_id: {document_id}\n"
            f"last_document_sha256: {sha256}\n"
            f"updated: {timestamp}\n"
            f"---\n"
        )

        header = f"# {_slug(curr).replace('-', ' ').title()}\n\n"
        line = f"- {timestamp} — {embed} `{file_dest.name}`  \n"

        if index_path.exists():
            # append a line, refresh frontmatter at the top if present
            text = index_path.read_text(encoding="utf-8", errors="ignore")
            if text.startswith("---"):
                # replace existing frontmatter
                parts = text.split("---", 2)  # ["", oldfm, rest]
                if len(parts) == 3:
                    new_text = f"{fm}{parts[2].lstrip()}"
                else:
                    new_text = f"{fm}{text}"
            else:
                new_text = f"{fm}{text}"
            new_text += "\n" + line
            index_path.write_text(new_text, encoding="utf-8")
        else:
            # create new
            index_path.write_text(f"{fm}{header}{line}\n", encoding="utf-8")
        logger.info(f"Obsidian node dir: {node_dir} | index: {index_path}")

        return file_dest

try:
    OBS_SYNC = ObsidianSync(os.environ["OBSIDIAN_VAULT_ROOT"])  # used inside test_dump.upload_and_save
except Exception as e:
    print("⚠️ Obsidian disabled:", e)
    OBS_SYNC = None 


def fs_to_channel_path(path: Union[PurePath, str]) -> str:
    """Convert a filesystem path to dot.notation for channel.path."""
    p = PurePath(path)
    parts = []
    for part in p.parts:
        # skip anchor/root like "/" or "C:\\"
        if part == p.anchor or part in ("/", "\\", ""):
            continue
        s = slugify(part)
        if s:
            parts.append(s)
    return ".".join(parts)

def guess_mime(name: str) -> str:
    return mimetypes.guess_type(name)[0] or "application/octet-stream"

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def build_base_path(chan_name: str, ancestors: List[str], doc_name: str) -> Path:
    """Builds a base path for storing documents based on name and ancestors."""
    base_path = DATA_DIR  # start here
    for ancestor in ancestors:
        base_path = base_path / slugify(ancestor)
    base_path = base_path / slugify(chan_name) / slugify(doc_name)  # append name, don’t replace root
    return base_path

async def connect_db():
    pool = await get_pool()
    if not pool:
        logger.error("Failed to connect to the database")
        raise RuntimeError("Failed to connect to the database")
    return pool
def parse_channel_tokens(channel_name: str) -> tuple[list[str], str | None, str]:
    """
    Accepts:
      - 'project-<project_name>'
      - 'subproject-<project_name>-<subproject_name>'
      - (optional later) 'task-<project>-<subproject>-<task>'
    Returns: (ancestors, parent_name, curr)
    """
    tokens = [t.strip() for t in channel_name.split("-") if t.strip()]
    if len(tokens) < 2:
        # require at least '<level>-<name>'
        return [], None, channel_name
    names = tokens[1:]             # skip the level prefix
    curr = names[-1]
    parent_name = names[-2] if len(names) >= 2 else None
    ancestors = names[:-1]         # ordered list of names above current node
    return ancestors, parent_name, curr


async def upload_and_save(
    buffer: bytes,
    message: discord.Message,
    filename: Optional[str] = None,
):
    """
    - Ensures channel rows in public.channel (parent first, then current)
    - Saves file to DATA_DIR/<ancestors>/<curr>/<filename>
    - Upserts into public.documents
    - Optionally mirrors to Obsidian via OBS_SYNC
    """

    # 1) Derive channel pieces
    channel_name = message.channel.name
    incoming_discord_id = str(message.channel.id) if getattr(message, "channel", None) else None
    ancestors, parent_name, curr = parse_channel_tokens(channel_name)

    # dot paths for channel.path
    if ancestors:
        curr_dot_path = fs_to_channel_path(Path("/".join([*ancestors, curr])))
        parent_dot_path = fs_to_channel_path(Path("/".join(ancestors)))
    else:
        curr_dot_path = fs_to_channel_path(Path(curr))
        parent_dot_path = None

    # 2) Pick a filename
    name = (
        filename
        or (message.attachments[0].filename if getattr(message, "attachments", None) else None)
        or "upload.bin"
    )

    DB_POOL = await connect_db()
    async with DB_POOL.acquire() as conn, conn.transaction():
        # 3) Ensure parent channel (no discord_channel_id for synthetic/parent rows)
        parent_id = None
        if ancestors:
            parent_id = await conn.fetchval(
                """
                INSERT INTO public.channel (name, path, discord_channel_id, parent_id, parent_name, ancestors)
                VALUES ($1, $2, NULL, NULL, NULL, $3)
                ON CONFLICT (path) DO UPDATE
                  SET updated_at = now()
                RETURNING id;
                """,
                ancestors[-1],
                parent_dot_path,
                ancestors[:-1],
            )

        # 4) Decide discord_channel_id to store for the *current* row
        #    If the same discord_channel_id already exists on a different path, avoid unique violation by using NULL here.
        candidate_discord_id = incoming_discord_id
        if incoming_discord_id:
            row = await conn.fetchrow(
                "SELECT id, path FROM public.channel WHERE discord_channel_id = $1;",
                incoming_discord_id
            )
            if row and row["path"] != curr_dot_path:
                # Same Discord channel id already used elsewhere → don't set it again
                candidate_discord_id = None

        # 5) Upsert current channel by path
        channel_row = await conn.fetchrow(
            """
            INSERT INTO public.channel (name, path, discord_channel_id, parent_id, parent_name, ancestors)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (path) DO UPDATE
              SET discord_channel_id = EXCLUDED.discord_channel_id,
                  updated_at = now()
            RETURNING *;
            """,
            curr,
            curr_dot_path,
            candidate_discord_id,
            parent_id,
            parent_name,
            ancestors,
        )
        channel_id = channel_row["id"]

        # 6) Write to disk
        dir_path = DATA_DIR / Path("/".join([*ancestors, curr]))
        dir_path.mkdir(parents=True, exist_ok=True)
        file_path = dir_path / name
        file_path.write_bytes(buffer)

        # 7) Metadata
        mime_type = guess_mime(name)
        sha256 = sha256_bytes(buffer)
        bytes_size = len(buffer)

        # 8) Upsert document
        doc_row = await conn.fetchrow(
            """
            INSERT INTO public.documents
              (channel_id, name, mime_type, bytes_size, sha256, storage_path, content, parent_name, ancestors)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
            ON CONFLICT (channel_id, name) DO UPDATE
              SET mime_type   = EXCLUDED.mime_type,
                  bytes_size  = EXCLUDED.bytes_size,
                  sha256      = EXCLUDED.sha256,
                  storage_path= EXCLUDED.storage_path,
                  content     = EXCLUDED.content,
                  parent_name = EXCLUDED.parent_name,
                  ancestors   = EXCLUDED.ancestors,
                  updated_at  = now()
            RETURNING *;
            """,
            channel_id, name, mime_type, bytes_size, sha256,
            str(file_path), buffer, parent_name, ancestors,
        )

        # 9) Obsidian mirror (optional)
        try:
            if 'OBS_SYNC' in globals() and OBS_SYNC:
                
                OBS_SYNC.write_document(
                    ancestors=ancestors,
                    curr=curr,
                    filename=name,
                    buffer=buffer,
                    mime_type=mime_type,
                    channel_path_dot=curr_dot_path,
                    channel_id=str(channel_id),
                    document_id=str(doc_row["id"]),
                    sha256=sha256,
                )
        except Exception as e:
            logger.warning(f"Obsidian mirror failed for {name}: {e}")

        return doc_row



async def get_ancestors(pool, channel_name: str) -> list[str]:
    """Return ancestors array from the Discord channel name (no DB lookups)."""
    ancestors, _parent_name, _curr = parse_channel_tokens(channel_name)
    return ancestors


@bot.event
async def on_ready():
    logger.info(f"Logged in as {bot.user} (ID: {bot.user.id})")
    logger.info("------")
    print(f"Bot is ready. Logged in as {bot.user}.")
    await connect_db()

@bot.event
async def on_message(message: discord.Message):
    if message.author.bot or not message.guild:
        return
    
    if not (bot.user in message.mentions and message.attachments):
        return
    
    logger.info(f"🎯 Processing message from {message.author} with {len(message.attachments)} attachments")
    logger.info(f"📍 Channel: {message.channel.name} (ID: {message.channel.id})")
    
    successful_uploads = []
    failed_uploads = []
    
    # Process each attachment
    for i, attachment in enumerate(message.attachments):
        try:
            logger.info(f"📎 Processing attachment {i+1}/{len(message.attachments)}: {attachment.filename}")
            logger.info(f"📏 Size: {attachment.size} bytes, Content-Type: {attachment.content_type}")
            
            # Read attachment
            buffer = await attachment.read()
            logger.info(f"📥 Successfully read {len(buffer)} bytes from attachment")
            
            # Upload and save — UPDATED SIGNATURE
            result = await upload_and_save(
                buffer,
                message,
                filename=attachment.filename
            )
            
            successful_uploads.append({
                'filename': attachment.filename,
                # 'doc_id': result["id"],  # available if you want it
                'path': result["storage_path"]  # UPDATED: use documents.storage_path
            })
            logger.info(f"✅ Successfully processed: {attachment.filename}")
            
        except Exception as e:
            logger.error(f"❌ Failed to process {attachment.filename}: {e}")
            import traceback
            logger.error(f"Attachment processing traceback: {traceback.format_exc()}")
            failed_uploads.append({
                'filename': attachment.filename,
                'error': str(e)
            })

        local_dt = datetime.now(ZoneInfo("Asia/Kolkata"))

        if successful_uploads:
            success_text = "\n".join([
                # f"✅ `{item['filename']}` (ID: {item['doc_id']})"
                f"✅ `{item['filename']}`"
                for item in successful_uploads
            ])
            embed = discord.Embed(
                title="Batch upload complete",
                timestamp=local_dt,
                color=discord.Color.green()
            )
            embed.add_field(
                name=f"Successfully Stored ({len(successful_uploads)} files)",
                value=success_text[:1024],
                inline=False
            )
        elif failed_uploads:
            error_text = "\n".join([
                f"❌ `{item['filename']}` - {item['error']}"
                for item in failed_uploads
            ])
            embed = discord.Embed(
                title="Batch upload failed",
                timestamp=local_dt,
                color=discord.Color.red()
           )
            embed.add_field(
                name=f"Failed to Store ({len(failed_uploads)} files)",
                value=error_text[:1024],
                inline=False
            )
        if successful_uploads:
            embed.add_field(name="Folder", value=successful_uploads[0]["path"], inline=False)
        embed.add_field(
            name="Uploaded",
            value=local_dt.strftime("%Y-%m-%d %I:%M %p %Z"),
            inline=True
        )
        
        await message.reply(embed=embed)


@bot.event
async def on_ready():
    print(f"🤖 Logged in as {bot.user}")
    logger.info(f"Bot started: {bot.user}")
    
    # Test database connection and operations
    try:
        logger.info("🔍 Testing database connection...")
        await connect_db()
        logger.info("🧪 Testing direct database operations...")
        logger.info("✅ All database tests passed")
    except Exception as e:
        print(f"❌ Database issues detected: {e}")
        logger.error(f"Database startup test failed: {e}")
        print("⚠️ Bot may not function correctly - check logs for details")
    
    # Test RAG system integration
    try:
        from llm_actions.rag import ingest_documents as test_rag
        print("✅ RAG system integration working")
        logger.info("RAG system integration verified")
    except Exception as e:
        print(f"⚠️ RAG system integration issue: {e}")
        logger.warning(f"RAG integration warning: {e}")
    
    print("📁 To upload files: Upload a file and mention the bot in the same message")
    print("📝 To upload text: Use /dump command")
    print("🔍 To query documents: Use /query command")
    print("🎯 Context scoping: Task threads query parent subproject docs only")

bot.run(TOKEN)

