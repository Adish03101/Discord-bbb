import os
import io
import re
import json
import mimetypes
import hashlib
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePath
from typing import Optional, List, Union, Any
from zoneinfo import ZoneInfo

# ==== Minimal slugify fallback ====
def simple_slugify(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"[^a-z0-9\-]+", "-", t)
    t = re.sub(r"-{2,}", "-", t).strip("-")
    return t

# ==== Configure logging for test harness ====
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
logger = logging.getLogger("offline_harness")

# ==== Constants similar to your script ====
ATTACHMENTS_DIR = "attachments"
INDEX_NAME = "index.md"

# ==== Implementations copied/adapted from your script ====

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
        node_dir = self._node_dir(ancestors, curr)
        attachments_dir = node_dir / ATTACHMENTS_DIR
        file_dest = _safe_filename(attachments_dir, filename)
        file_dest.write_bytes(buffer)

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
            text = index_path.read_text(encoding="utf-8", errors="ignore")
            if text.startswith("---"):
                parts = text.split("---", 2)
                if len(parts) == 3:
                    new_text = f"{fm}{parts[2].lstrip()}"
                else:
                    new_text = f"{fm}{text}"
            else:
                new_text = f"{fm}{text}"
            new_text += "\n" + line
            index_path.write_text(new_text, encoding="utf-8")
        else:
            index_path.write_text(f"{fm}{header}{line}\n", encoding="utf-8")

        return file_dest

def fs_to_channel_path(path: Union[PurePath, str]) -> str:
    p = PurePath(path)
    parts = []
    for part in p.parts:
        if part == p.anchor or part in ("/", "\\", ""):
            continue
        s = simple_slugify(part)   # using our fallback
        if s:
            parts.append(s)
    return ".".join(parts)

def guess_mime(name: str) -> str:
    return mimetypes.guess_type(name)[0] or "application/octet-stream"

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def parse_channel_tokens(channel_name: str) -> tuple[list[str], str | None, str]:
    tokens = [t.strip() for t in channel_name.split("-") if t.strip()]
    if len(tokens) < 2:
        return [], None, channel_name
    level = tokens[0]
    names = tokens[1:]
    if level == "project":
        return [], None, names[0]
    elif level == "subproject" and len(names) >= 2:
        return names[:-1], names[-2], names[-1]
    else:
        curr = names[-1] if names else channel_name
        parent_name = names[-2] if len(names) >= 2 else None
        ancestors = names[:-1] if len(names) > 1 else []
        return ancestors, parent_name, curr

# ==== Fakes for Discord + DB ====

@dataclass
class FakeAttachment:
    filename: str
    size: int
    content_type: str
    data: bytes
    async def read(self) -> bytes:
        return self.data

@dataclass
class FakeChannel:
    name: str
    id: int

@dataclass
class FakeAuthor:
    bot: bool = False

@dataclass
class FakeGuild:
    id: int = 1

@dataclass
class FakeMessage:
    channel: FakeChannel
    attachments: List[FakeAttachment] = field(default_factory=list)
    author: FakeAuthor = field(default_factory=FakeAuthor)
    guild: Optional[FakeGuild] = field(default_factory=FakeGuild)

# Fake DB that records operations and returns deterministic ids/rows
class FakeConnection:
    def __init__(self, store: dict):
        self.store = store
    async def __aenter__(self):
        return self
    async def __aexit__(self, exc_type, exc, tb):
        pass
    async def transaction(self):
        # minimal context manager for async with
        class _Txn:
            async def __aenter__(self_inner): return None
            async def __aexit__(self_inner, *args): return False
        return _Txn()
    async def fetchval(self, query: str, *params):
        # Create or fetch parent channel id
        path = params[1]
        parent_id = self.store["channels"].get(path, {"id": len(self.store["channels"])+1}).get("id")
        self.store["channels"][path] = {
            "id": parent_id,
            "name": params[0],
            "path": path,
            "ancestors": params[2],
            "discord_channel_id": None,
        }
        return parent_id
    async def fetchrow(self, query: str, *params):
        if "INSERT INTO public.channel" in query:
            name, path, discord_channel_id, parent_id, parent_name, ancestors = params
            row = self.store["channels"].get(path, {"id": len(self.store["channels"])+1})
            row.update({
                "name": name, "path": path, "discord_channel_id": discord_channel_id,
                "parent_id": parent_id, "parent_name": parent_name, "ancestors": ancestors,
                "updated_at": datetime.now(timezone.utc),
                "id": row.get("id", len(self.store["channels"])+1)
            })
            self.store["channels"][path] = row
            return row
        elif "INSERT INTO public.documents" in query:
            channel_id, name, mime_type, bytes_size, sha256, storage_path, content, parent_name, ancestors = params
            doc_id = len(self.store["documents"])+1
            row = {
                "id": doc_id, "channel_id": channel_id, "name": name, "mime_type": mime_type,
                "bytes_size": bytes_size, "sha256": sha256, "storage_path": storage_path,
                "content": content, "parent_name": parent_name, "ancestors": ancestors,
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc)
            }
            self.store["documents"][doc_id] = row
            return row
        else:
            raise NotImplementedError("Query not recognized in fake DB")

class FakePool:
    def __init__(self, store: dict):
        self.store = store
    async def acquire(self):
        return FakeConnection(self.store)

# We'll emulate `connect_db()` using our FakePool
FAKE_DB_STORE = {"channels": {}, "documents": {}}
_fake_pool = FakePool(FAKE_DB_STORE)

async def connect_db():
    return _fake_pool

# ==== upload_and_save adapted to use our connect_db + ObsidianSync injected ====

DATA_DIR = Path("/mnt/data/offline_data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Prepare a temp Obsidian vault
VAULT_DIR = Path("/mnt/data/obsidian_vault")
os.environ["OBSIDIAN_VAULT_ROOT"] = str(VAULT_DIR)

OBS_SYNC = ObsidianSync()

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def guess_mime(name: str) -> str:
    return mimetypes.guess_type(name)[0] or "application/octet-stream"

def fs_to_channel_path(path: Union[PurePath, str]) -> str:
    p = PurePath(path)
    parts = []
    for part in p.parts:
        if part == p.anchor or part in ("/", "\\", ""):
            continue
        s = simple_slugify(part)
        if s:
            parts.append(s)
    return ".".join(parts)

async def upload_and_save(
    buffer: bytes,
    message: FakeMessage,
    filename: Optional[str] = None,
):
    channel_name = message.channel.name
    discord_channel_id = str(message.channel.id)
    ancestors, parent_name, curr = parse_channel_tokens(channel_name)
    if ancestors:
        curr_dot_path = fs_to_channel_path(Path("/".join([*ancestors, curr])))
        parent_dot_path = fs_to_channel_path(Path("/".join(ancestors)))
    else:
        curr_dot_path = fs_to_channel_path(Path(curr))
        parent_dot_path = None

    name = filename or (message.attachments[0].filename if getattr(message, "attachments", None) else None) or "upload.bin"

    pool = await connect_db()
    async with pool.acquire() as conn:
        # parent
        parent_id = None
        if ancestors and parent_dot_path:
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
        # current channel
        channel_row = await conn.fetchrow(
            """
            INSERT INTO public.channel (name, path, discord_channel_id, parent_id, parent_name, ancestors)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (path) DO UPDATE
              SET discord_channel_id = EXCLUDED.discord_channel_id,
                  updated_at = now()
            RETURNING *;
            """,
            curr, curr_dot_path, discord_channel_id, parent_id, parent_name, ancestors,
        )
        channel_id = channel_row["id"]

        # write file
        dir_path = DATA_DIR
        for ancestor in ancestors:
            dir_path = dir_path / simple_slugify(ancestor)
        dir_path = dir_path / simple_slugify(curr)
        dir_path.mkdir(parents=True, exist_ok=True)
        file_path = dir_path / name
        file_path.write_bytes(buffer)

        # metadata
        mime_type = guess_mime(name)
        sha256 = sha256_bytes(buffer)
        bytes_size = len(buffer)

        # document row
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

        # Obsidian mirror
        if OBS_SYNC:
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
                created_at_utc=doc_row["created_at"],
            )
        return {
            "doc_row": doc_row,
            "file_path": str(file_path),
            "channel_row": channel_row,
            "ancestors": ancestors,
            "curr": curr,
            "curr_dot_path": curr_dot_path,
        }

# ==== Unit-style tests for each function ====

results = {}

# 1) _slug
results["_slug"] = [
    _slug(" Hello  World  "),
    _slug("Project@Name!"),
    _slug(""),
]

# 2) _safe_filename
tmp_dir = Path("/mnt/data/tmp_safe")
tmp_dir.mkdir(parents=True, exist_ok=True)
p1 = _safe_filename(tmp_dir, "file.txt"); p1.write_text("x")
p2 = _safe_filename(tmp_dir, "file.txt"); p2.write_text("y")
p3 = _safe_filename(tmp_dir, "file.txt"); p3.write_text("z")
results["_safe_filename"] = [p1.name, p2.name, p3.name]

# 3) _embed_line
results["_embed_line"] = [
    _embed_line("image/png", "attachments/img.png"),
    _embed_line("application/pdf", "attachments/paper.pdf"),
    _embed_line("text/plain", "attachments/readme.txt"),
]

# 4) fs_to_channel_path
results["fs_to_channel_path"] = [
    fs_to_channel_path(PurePath("/alpha/beta/gamma")),
    fs_to_channel_path("Project X/Sub One"),
    fs_to_channel_path("C:\\Root\\A\\B"),
]

# 5) guess_mime
results["guess_mime"] = [
    guess_mime("a.png"),
    guess_mime("b.mp3"),
    guess_mime("c.unknownext"),
]

# 6) sha256_bytes
results["sha256_bytes"] = sha256_bytes(b"hello world")

# 7) parse_channel_tokens
results["parse_channel_tokens"] = [
    parse_channel_tokens("project-uno"),
    parse_channel_tokens("subproject-uno-dos"),
    parse_channel_tokens("weird-channel-name"),
]

# 8) ObsidianSync._node_dir + write_document
test_vault = VAULT_DIR
anc = ["Uno"]
cur = "Dos"
obs = ObsidianSync(str(test_vault))
node = obs._node_dir(anc, cur)
# write a file
obs_written = obs.write_document(
    ancestors=anc, curr=cur, filename="note.txt", buffer=b"data",
    mime_type="text/plain", channel_path_dot="uno.dos", channel_id="1",
    document_id="42", sha256=sha256_bytes(b"data")
)
results["obsidian_paths"] = [str(node), str(obs_written)]
results["obsidian_index_exists"] = (node / INDEX_NAME).exists()

# 9) upload_and_save (integration)
fake_msg = FakeMessage(
    channel=FakeChannel(name="subproject-uno-dos", id=777),
    attachments=[FakeAttachment(filename="report.md", size=11, content_type="text/markdown", data=b"# Title\nHello")]
)
import asyncio
integration_out = asyncio.get_event_loop().run_until_complete(
    upload_and_save(buffer=b"# Title\nHello", message=fake_msg, filename="report.md")
)
results["upload_and_save"] = {
    "file_path": integration_out["file_path"],
    "channel_path": integration_out["curr_dot_path"],
    "ancestors": integration_out["ancestors"],
    "curr": integration_out["curr"],
    "db_channels": list(FAKE_DB_STORE["channels"].keys()),
    "db_documents_count": len(FAKE_DB_STORE["documents"]),
    "obsidian_index": str((test_vault / "uno" / "dos" / "index.md")),
    "obsidian_index_exists": (test_vault / "uno" / "dos" / "index.md").exists(),
}

# Display summary JSON
print(json.dumps(results, indent=2))
