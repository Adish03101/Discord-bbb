import os
import asyncio
import logging
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime, timezone
import sys
import discord
import io
from discord.commands import Option
from dotenv import load_dotenv
from slugify import slugify
import uuid
from typing import Optional
sys.path.append("/home/black/Backup/Discord-bbb/project-root/src")

# RAG and database helpers (ensure project is installed as a package)
from llm_actions.rag import fetch_texts_from_subproject, ingest_documents, rag_generate
from llm_actions.summarization import get_summarizer
from core.pg_db import get_pool, upsert_document

# ——— CONFIGURATION —————————————————————————————————————————————
load_dotenv()
DATABASE_URL = os.getenv("POSTGRES_URL")
RAG_INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", "./indices"))
DATA_DIR = Path(os.getenv("STORAGE_DIR", "./data/files"))
TOKEN = os.getenv("TASK_TOKEN")

# Ensure storage directories exist
RAG_INDEX_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("task_bot")

# Discord bot setup
intents = discord.Intents.default()
intents.message_content = True
bot = discord.Bot(intents=intents)

# Concurrency guards for index builds
tasks_locks: Dict[str, asyncio.Lock] = {}

# ——— SCHEMA MIGRATION AND DATABASE SETUP —————————————————————————
async def ensure_schema():
    """Ensure database schema and UUID extension are properly set up"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        try:
            # First, ensure uuid-ossp extension is available
            await conn.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
            
            # Check existing columns
            cols = await conn.fetch(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'channels'
                AND column_name IN ('created_at','status','task_name')
                """
            )
            existing = {r['column_name'] for r in cols}
            
            # Add missing columns
            if 'created_at' not in existing:
                await conn.execute(
                    "ALTER TABLE channels ADD COLUMN created_at TIMESTAMPTZ NOT NULL DEFAULT now()"
                )
                logger.info("Added created_at column to channels table")
                
            if 'status' not in existing:
                await conn.execute(
                    "ALTER TABLE channels ADD COLUMN status TEXT NOT NULL DEFAULT 'active'"
                )
                logger.info("Added status column to channels table")
                
            if 'task_name' not in existing:
                await conn.execute(
                    "ALTER TABLE channels ADD COLUMN task_name TEXT"
                )
                logger.info("Added task_name column to channels table")
                
        except Exception as e:
            logger.error(f"Error setting up database schema: {e}")
            raise

# ——— DATA ACCESS LAYER —————————————————————————————————————————

async def get_subproject_doc_id_for_thread(thread: discord.Thread) -> Optional[str]:
    """Get the parent subproject's document ID for a thread"""
    if not isinstance(thread.parent, discord.TextChannel):
        return None
    
    parent_channel = thread.parent
    # Verify it's a subproject channel
    if not parent_channel.name.startswith('subproject-'):
        return None
    
    return await get_channel_doc_id(parent_channel)

async def get_channel_doc_id(channel: discord.abc.GuildChannel) -> Optional[str]:
    """Get document ID for a channel, handling UUID conversion properly"""
    chan_id = str(channel.id)
    pool = await get_pool()
    
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT document_id FROM channels WHERE channel_id = $1", chan_id
            )
            
            if row and row['document_id']:
                # Convert UUID to string if necessary
                doc_id = row['document_id']
                return str(doc_id)
            
            # Handle subproject auto-registration
            if channel.name.startswith('subproject-'):
                try:
                    from dump import parse_channel_name, build_base_path
                    level, own_slug, parent_slug = parse_channel_name(channel.name)
                    
                    doc_res = await upsert_document(
                        pool,
                        original_name=channel.name,
                        slug=f"{level}-{own_slug}",
                        parent_id=None,
                        content_type=None,
                        file_bytes=None,
                        summary=None,
                        channel_id=chan_id,
                    )
                    
                    doc_id = str(doc_res['id'])
                    base_path = build_base_path(level, own_slug, None)
                    base_path.mkdir(parents=True, exist_ok=True)
                    
                    # Convert doc_id string to UUID for database insertion
                    doc_uuid = uuid.UUID(doc_id)
                    
                    await conn.execute(
                        """
                        INSERT INTO channels
                        (channel_id, document_id, ancestors, path, created_at, status, task_name)
                        VALUES ($1,$2,$3,$4,$5,$6,$7)
                        """,
                        chan_id, doc_uuid, [], str(base_path),
                        datetime.now(timezone.utc), 'active', own_slug
                    )
                    
                    return doc_id
                    
                except Exception as e:
                    logger.exception(f"Failed to auto-register subproject {channel.name}: {e}")
                    
    except Exception as e:
        logger.error(f"Error getting channel doc ID: {e}")
    
    return None

async def list_active_tasks_ids(sub_id: str) -> List[Dict]:
    """List active task IDs with proper error handling"""
    if not sub_id:
        return []
    
    pool = await get_pool()
    try:
        async with pool.acquire() as conn:
            # Use string parameters and let PostgreSQL handle the conversion
            rows = await conn.fetch(
                """
                SELECT c.channel_id, c.task_name
                FROM channels c
                JOIN documents d ON c.document_id = d.id
                WHERE c.status = 'active'
                AND (d.parent_id::text = $1 OR $1 = ANY(c.ancestors::text[]))
                """,
                sub_id
            )
            
            return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Error listing active tasks: {e}")
        return []

async def task_exists(name: str, sub_id: str) -> bool:
    """Check if task exists with proper return statement"""
    if not name or not sub_id:
        return False
    
    pool = await get_pool()
    try:
        async with pool.acquire() as conn:
            exists = await conn.fetchval(
                """
                SELECT 1
                FROM channels c
                JOIN documents d ON c.document_id = d.id
                WHERE c.task_name = $1
                AND c.status = 'active'
                AND (d.parent_id::text = $2 OR $2 = ANY(c.ancestors::text[]))
                """,
                name, sub_id
            )
            return exists is not None
    except Exception as e:
        logger.error(f"Error checking if task exists: {e}")
        return False

async def get_latest_uploaded_at(root_id: str) -> Optional[datetime]:
    """Get latest upload time with proper error handling"""
    if not root_id:
        return None
    
    pool = await get_pool()
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT MAX(uploaded_at) AS latest FROM documents WHERE parent_id::text = $1",
                root_id
            )
            return row['latest'] if row and row['latest'] else None
    except Exception as e:
        logger.error(f"Error getting latest upload time: {e}")
        return None

# ——— INDEXING UTILITY —————————————————————————————————————————

async def build_or_refresh_index(subproject_doc_id: str):
    """Build or refresh index with improved error handling - only for subproject docs"""
    if not subproject_doc_id:
        logger.warning("Cannot build index: subproject_doc_id is None")
        return
    
    lock = tasks_locks.setdefault(subproject_doc_id, asyncio.Lock())
    async with lock:
        try:
            idx_path = RAG_INDEX_DIR / f"{subproject_doc_id}.index"
            rebuild = True
            
            if idx_path.exists():
                latest = await get_latest_uploaded_at(subproject_doc_id)
                if latest and idx_path.stat().st_mtime >= latest.timestamp():
                    logger.debug("Index for %s is current", subproject_doc_id)
                    rebuild = False
            
            if not rebuild:
                return
            
            # FIXED: Use the new function that only fetches subproject documents
            texts = await fetch_texts_from_subproject(subproject_doc_id, DATABASE_URL)
            
            if not texts:
                logger.warning("No docs to index for %s; skipping", subproject_doc_id)
                return
            
            await asyncio.to_thread(ingest_documents, subproject_doc_id, texts)
            logger.info("Indexed %d docs for subproject %s", len(texts), subproject_doc_id)
            
        except Exception as e:
            logger.exception(f"Index build failed for {subproject_doc_id}: {e}")
            raise

# ——— EVENT HANDLERS —————————————————————————————————————————

@bot.event
async def on_ready():
    """Initialize bot with proper error handling"""
    try:
        await ensure_schema()
        logger.info("Task Bot logged in as %s", bot.user)
    except Exception as e:
        logger.error(f"Failed to initialize bot: {e}")
        raise

@bot.event
async def on_thread_update(before: discord.Thread, after: discord.Thread):
    """Handle thread updates with error handling"""
    if before.name != after.name:
        try:
            pool = await get_pool()
            new_slug = after.name.replace("task-", "", 1)
            
            async with pool.acquire() as conn:
                await conn.execute(
                    "UPDATE channels SET task_name = $1 WHERE channel_id = $2",
                    slugify(new_slug), str(after.id)
                )
                
            logger.info(f"Updated task name for thread {after.id}")
        except Exception as e:
            logger.error(f"Error updating thread name: {e}")

# ——— SLASH COMMANDS —————————————————————————————————————————

@bot.slash_command(name="task_query", description="Ask a question about files in this subproject")
async def task_query(
    ctx: discord.ApplicationContext,
    query: Option(str, "Your question")
):
    deferred = False
    try:
        # 0) Try to defer
        try:
            await ctx.defer()
            deferred = True
        except discord.errors.NotFound:
            logger.warning("Could not defer interaction; it may have expired.")
        except Exception:
            logger.exception("Unexpected error during ctx.defer()")

        # 1) Resolve the correct document ID
        if isinstance(ctx.channel, discord.Thread):
            doc_id = await get_subproject_doc_id_for_thread(ctx.channel)
        else:
            doc_id = await get_channel_doc_id(ctx.channel)

        if not doc_id:
            msg = "⚠️ Not a subproject channel or thread."
            return await (ctx.followup.send if deferred else ctx.respond)(msg, ephemeral=True)

        # 2) Ensure the index exists / is up to date
        try:
            await asyncio.to_thread(build_or_refresh_index, doc_id)
        except Exception as e:
            logger.exception("Index build/refresh failed for doc_id=%s", doc_id)
            msg = f"⚠️ Failed to build index: {e}"
            return await (ctx.followup.send if deferred else ctx.respond)(msg, ephemeral=True)

        # 3) Perform RAG  
        try:
            top_k = int(os.getenv("RAG_TOP_K", "4"))
            answer = await asyncio.to_thread(rag_generate, doc_id, query, top_k)
        except Exception as e:
            logger.exception("RAG generation failed for doc_id=%s query=%s", doc_id, query)
            msg = f"⚠️ Query generation failed: {e}"
            return await (ctx.followup.send if deferred else ctx.respond)(msg, ephemeral=True)

        # 4) Assemble and respect Discord limits
        header = f"**Question:** {query}\n\n**Answer:**\n"
        content = header + answer

        if len(content) <= 2000:
            return await (ctx.followup.send if deferred else ctx.respond)(content)
        else:
            md = f"# Question\n{query}\n\n# Answer\n{answer}"
            buf = io.BytesIO(md.encode("utf-8"))
            file = discord.File(buf, filename="answer.md")
            notice = "⚠️ Answer was too long, sending as a markdown file."
            return await (ctx.followup.send if deferred else ctx.respond)(notice, file=file)

    except Exception:
        logger.exception("Unhandled exception in task_query")
        fallback = "❌ An unexpected error occurred. Check logs."
        return await (ctx.followup.send if deferred else ctx.respond)(fallback, ephemeral=True)


@bot.slash_command(name='list_active_tasks', description='List active tasks in this subproject')
async def list_active_tasks(ctx: discord.ApplicationContext):
    """List active tasks with improved error handling"""
    await ctx.defer(ephemeral=True)
    
    try:
        # Handle both direct subproject channels and threads
        if isinstance(ctx.channel, discord.Thread):
            subproject_doc_id = await get_subproject_doc_id_for_thread(ctx.channel)
        else:
            subproject_doc_id = await get_channel_doc_id(ctx.channel)
            
        if not subproject_doc_id:
            return await ctx.respond('⚠️ Not in a subproject.', ephemeral=True)
        
        rows = await list_active_tasks_ids(subproject_doc_id)
        
        if not rows:
            return await ctx.respond('✅ No active tasks.', ephemeral=True)
        
        lines = [f"• <#{r['channel_id']}> — `{r['task_name'] or 'unnamed'}`" for r in rows]
        await ctx.respond('**Active Tasks:**\n' + '\n'.join(lines), ephemeral=True)
        
    except Exception as e:
        logger.exception(f"list_active_tasks failed: {e}")
        await ctx.respond("❌ An error occurred listing tasks.", ephemeral=True)

@bot.slash_command(name='create_task', description='Create a new task thread under this subproject')
async def create_task(
    ctx: discord.ApplicationContext,
    name: Option(str, 'Short name for the new task')
):
    """Create task with improved validation and error handling"""
    await ctx.defer(ephemeral=True)
    
    try:
        if isinstance(ctx.channel, discord.Thread):
            return await ctx.respond('⚠️ Run in parent subproject, not a thread.', ephemeral=True)
        
        chan = ctx.channel
        sub_id = await get_channel_doc_id(chan)
        
        if not sub_id:
            return await ctx.respond('⚠️ Not in a subproject.', ephemeral=True)
        
        # Validate name
        if not name or not name.strip():
            return await ctx.respond('⚠️ Task name cannot be empty.', ephemeral=True)
        
        slug = slugify(name)
        if not slug:
            return await ctx.respond('⚠️ Invalid task name.', ephemeral=True)
        
        if await task_exists(slug, sub_id):
            return await ctx.respond(f'⚠️ Task `{name}` already exists.', ephemeral=True)
        
        thread = await chan.create_thread(name=f'task-{slug}', auto_archive_duration=1440)
        
        pool = await get_pool()
        async with pool.acquire() as conn:
            parent_info = await conn.fetchrow(
                "SELECT ancestors FROM channels WHERE channel_id = $1", str(chan.id)
            )
            
            # Convert sub_id to UUID for database operations, but keep ancestors as strings
            sub_uuid = uuid.UUID(sub_id)
            ancestors = list(parent_info.get('ancestors') or []) + [sub_id] if parent_info else [sub_id]
            
            await conn.execute(
                """
                INSERT INTO channels (channel_id, document_id, ancestors, path, created_at, status, task_name)
                VALUES ($1,$2,$3,$4,$5,$6,$7)
                """,
                str(thread.id), sub_uuid, ancestors,
                thread.name, datetime.now(timezone.utc), 'active', slug
            )
        
        await ctx.followup.send(f'✅ Created task {thread.mention}', ephemeral=True)
        logger.info(f"Created task thread: {thread.name} (ID: {thread.id})")
        
    except Exception as e:
        logger.exception(f"create_task failed: {e}")
        await ctx.respond("❌ An error occurred creating the task.", ephemeral=True)

@bot.slash_command(name='close_task', description='Mark this task thread as complete')
async def close_task(ctx: discord.ApplicationContext):
    """Close task with error handling"""
    try:
        if not isinstance(ctx.channel, discord.Thread):
            return await ctx.respond('⚠️ Use inside a task thread.', ephemeral=True)
        
        pool = await get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                "UPDATE channels SET status='complete' WHERE channel_id = $1",
                str(ctx.channel.id)
            )
        
        await ctx.respond('✅ Task marked complete.', ephemeral=True)
        logger.info(f"Closed task thread: {ctx.channel.id}")
        
    except Exception as e:
        logger.exception(f"close_task failed: {e}")
        await ctx.respond("❌ An error occurred closing the task.", ephemeral=True)

@bot.slash_command(name='summarize_task_thread', description='Summarize this thread and store under subproject')
async def summarize_task_thread(ctx: discord.ApplicationContext):
    """Summarize thread with improved error handling"""
    try:
        if not isinstance(ctx.channel, discord.Thread):
            return await ctx.respond('⚠️ Use inside a task thread.', ephemeral=True)
        
        parent = ctx.channel.parent
        sub_id = await get_channel_doc_id(parent)
        
        if not sub_id:
            return await ctx.respond('⚠️ Not a subproject.', ephemeral=True)
        
        # Collect messages
        msgs = [m async for m in ctx.channel.history(limit=1000, oldest_first=True)]
        texts: List[str] = []
        
        for m in msgs:
            parts: List[str] = []
            if m.content and m.content.strip():
                parts.append(m.content)
            
            for e in m.embeds:
                if e.title:
                    parts.append(f"📌 {e.title}")
                if e.description:
                    parts.append(e.description)
                if e.url:
                    parts.append(f"[Link]({e.url})")
            
            if parts:
                texts.append("\n".join(parts))
        
        if not texts:
            return await ctx.respond('⚠️ Nothing to summarize!', ephemeral=True)
        
        # Generate summary
        summary = await asyncio.to_thread(get_summarizer().summarize, texts)
        
        # Save summary
        folder = DATA_DIR / sub_id
        folder.mkdir(parents=True, exist_ok=True)
        fname = f"{ctx.channel.id}_summary.txt"
        fpath = folder / fname
        fpath.write_text(summary, encoding='utf-8')
        
        # Store in database
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO documents (blob_key, parent_id) VALUES ($1, $2)",
                f"{sub_id}/{fname}", uuid.UUID(sub_id)
            )
        
        # Update index
        await asyncio.to_thread(ingest_documents, sub_id, [summary])
        
        await ctx.respond('✅ Thread summary stored and indexed.', ephemeral=True)
        logger.info(f"Summarized thread: {ctx.channel.id}")
        
    except Exception as e:
        logger.exception(f"summarize_task_thread failed: {e}")
        await ctx.respond("❌ An error occurred summarizing the thread.", ephemeral=True)

@bot.slash_command(name='debug_db', description='Debug database contents for this channel')
async def debug_db(ctx: discord.ApplicationContext):
    """Debug database with improved error handling and parameterized queries"""
    await ctx.defer(ephemeral=True)
    
    try:
        if isinstance(ctx.channel, discord.Thread):
            chan = ctx.channel.parent
            subproject_doc_id = await get_subproject_doc_id_for_thread(ctx.channel)
        else:
            chan = ctx.channel
            subproject_doc_id = await get_channel_doc_id(chan)
            
        if not subproject_doc_id:
            return await ctx.respond('⚠️ Not in a subproject.', ephemeral=True)
        
        pool = await get_pool()
        async with pool.acquire() as conn:
            # Get channel info
            channel_info = await conn.fetchrow(
                "SELECT * FROM channels WHERE channel_id = $1", str(chan.id)
            )
            
            # Get documents for this subproject (parent_id)
            docs = await conn.fetch(
                """SELECT id, original_name, parent_id, content_type,
                   octet_length(file_data) AS size
                   FROM documents WHERE parent_id::text = $1""",
                subproject_doc_id
            )
            
            # Get all documents uploaded in this channel
            all_docs = await conn.fetch(
                """SELECT id, original_name, parent_id, content_type,
                   octet_length(file_data) AS size
                   FROM documents WHERE channel_id = $1""",
                str(chan.id)
            )
        
        # Build message lines
        lines = [
            f"**Channel Mapping:**",
            f"ID: {chan.id}",
            f"SubprojectDocID: {subproject_doc_id}",
            f"Info: {dict(channel_info) if channel_info else 'None'}",
            "",
            f"**Docs (parent_id={subproject_doc_id}):**",
        ]
        
        if docs:
            for d in docs:
                size_str = f"{d['size']} bytes" if d['size'] else "0 bytes"
                lines.append(f"• {d['original_name']} (ID={d['id']}, size={size_str})")
        else:
            lines.append("None")
        
        lines.append("")
        lines.append(f"**All docs in channel {chan.id}:**")
        
        if all_docs:
            for d in all_docs:
                size_str = f"{d['size']} bytes" if d['size'] else "0 bytes"
                lines.append(f"• {d['original_name']} (parent={d['parent_id']}, size={size_str})")
        else:
            lines.append("None")
        
        msg = "\n".join(lines)
        
        # Chunk and send (Discord has message limits)
        chunks = [msg[i:i+1900] for i in range(0, len(msg), 1900)]
        
        if chunks:
            await ctx.respond(chunks[0], ephemeral=True)
            for chunk in chunks[1:]:
                await ctx.followup.send(chunk, ephemeral=True)
        else:
            await ctx.respond('No debug info available.', ephemeral=True)
            
    except Exception as e:
        logger.exception(f"debug_db failed: {e}")
        await ctx.respond("❌ An error occurred during debug.", ephemeral=True)

# ——— MAIN EXECUTION —————————————————————————————————————————

if __name__ == '__main__':
    try:
        if not TOKEN:
            logger.error("TOKEN environment variable not set")
            sys.exit(1)
        if not DATABASE_URL:
            logger.error("DATABASE_URL environment variable not set")
            sys.exit(1)
        
        logger.info("Starting Discord Task Bot...")
        bot.run(TOKEN)
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
