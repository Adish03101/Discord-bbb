import os
import asyncio
import asyncpg
import hashlib

# Load environment if you use .env elsewhere
from dotenv import load_dotenv
load_dotenv()

# Connection URL, fallback to default if not provided
PG_URL = os.getenv("POSTGRES_URL", "postgresql://dumpbot:dumppass@localhost:5432/dumpdb")

_pool: asyncpg.Pool | None = None
_init_lock = asyncio.Lock()

async def get_pool():
    global _pool
    if _pool:
        return _pool
    async with _init_lock:
        if _pool:
            return _pool
        _pool = await asyncpg.create_pool(dsn=PG_URL, min_size=1, max_size=5)
        return _pool

async def upsert_document(
    pool,
    *,
    original_name: str,
    slug: str,
    parent_id: str | None = None,
    content_type: str | None = None,
    file_bytes: bytes | None = None,
    summary: str | None = None,
    channel_id: str | None = None,
):
    sha256 = hashlib.sha256(file_bytes).hexdigest() if file_bytes else None
    async with pool.acquire() as conn:
        if sha256:
            existing = await conn.fetchrow("SELECT id FROM documents WHERE hash=$1", sha256)
            if existing:
                return {"id": existing["id"], "reused": True}
        result = await conn.fetchrow(
            """
            INSERT INTO documents (original_name, slug, parent_id, content_type, file_size, file_data, summary, hash, channel_id)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
            RETURNING id
            """,
            original_name,
            slug,
            parent_id,
            content_type,
            len(file_bytes) if file_bytes else None,
            file_bytes,
            summary,
            sha256,
            channel_id,
        )
        return {"id": result["id"], "reused": False}

async def get_document_bytes(pool, doc_id: str) -> bytes | None:
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT file_data FROM documents WHERE id=$1", doc_id)
        return row["file_data"] if row else None
