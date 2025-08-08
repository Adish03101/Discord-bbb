import os
import sys
import asyncio
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import aiosqlite
import discord
from discord.commands import Option
import io
import logging

# make sure this path points at where rag.py lives
sys.path.append("/home/black/Backup/Discord-bbb/project-root/src")
from llm_actions.rag import ingest_documents, rag_generate

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("project_query_bot")

# Load environment variables
load_dotenv()
DB_FILE = os.getenv("DB_FILE", "./dumpbot.db")
DATA_DIR = Path(os.getenv("STORAGE_DIR", "./data/files"))
INDICES_DIR = Path(__file__).resolve().parent.parent / "indices"
TOKEN = os.getenv("PROJECT_TOKEN")

bot = discord.Bot(intents=discord.Intents.default())

async def get_project_root(channel_id: str) -> Optional[str]:
    """Retrieve the root document ID for this channel."""
    try:
        async with aiosqlite.connect(DB_FILE) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT document_id FROM channels WHERE channel_id = ?",
                (channel_id,),
            ) as cur:
                row = await cur.fetchone()
        return row["document_id"] if row else None
    except Exception:
        logger.exception("Failed to fetch project root for channel_id=%s", channel_id)
        return None

@bot.slash_command(name="project_query", description="Ask a question about this project")
async def project_query(ctx: discord.ApplicationContext,
                        query: Option(str, "Your question")):
    deferred = False
    try:
        try:
            await ctx.defer()
            deferred = True
        except discord.errors.NotFound:
            # interaction already invalid/expired; continue without defer
            logger.warning("Could not defer interaction; it may have expired.")
        except Exception:
            logger.exception("Unexpected error during ctx.defer()")

        root_id = await get_project_root(str(ctx.channel_id))
        if not root_id:
            msg = "⚠️ This channel isn't a project dump."
            if deferred:
                await ctx.followup.send(msg, ephemeral=True)
            else:
                await ctx.respond(msg, ephemeral=True)
            return

        # 1) Ingest documents if index doesn't exist yet
        idx_path = INDICES_DIR / root_id
        if not idx_path.exists():
            try:
                async with aiosqlite.connect(DB_FILE) as db:
                    db.row_factory = aiosqlite.Row
                    async with db.execute(
                        "SELECT blob_key FROM documents WHERE parent_id = ?",
                        (root_id,),
                    ) as cur:
                        rows = await cur.fetchall()
                texts = []
                for r in rows:
                    p = DATA_DIR / r["blob_key"]
                    try:
                        texts.append(p.read_text(encoding="utf-8", errors="ignore"))
                    except FileNotFoundError:
                        logger.warning("File not found for blob_key %s", r["blob_key"])
                        continue

                # Offload blocking ingestion to thread
                await asyncio.to_thread(ingest_documents, root_id, texts)

                # create sentinel so it doesn't re-ingest next time
                idx_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.exception("Ingestion failed for root_id=%s", root_id)
                msg = f"⚠️ Ingestion failed: {e}"
                if deferred:
                    await ctx.followup.send(msg, ephemeral=True)
                else:
                    await ctx.respond(msg, ephemeral=True)
                return

        # 2) Perform RAG query
        try:
            answer = await asyncio.to_thread(rag_generate, root_id, query, 4)
        except Exception as e:
            logger.exception("RAG generation failed for root_id=%s query=%s", root_id, query)
            msg = f"⚠️ Query generation failed: {e}"
            if deferred:
                await ctx.followup.send(msg, ephemeral=True)
            else:
                await ctx.respond(msg, ephemeral=True)
            return

        # 3) Respect Discord's limits
        header = f"**Question:** {query}\n\n**Answer:**\n"
        content = header + answer
        if len(content) <= 2000:
            if deferred:
                await ctx.followup.send(content)
            else:
                await ctx.respond(content)
        else:
            md_content = f"# Question\n{query}\n\n# Answer\n{answer}"
            buffer = io.BytesIO(md_content.encode("utf-8"))
            file = discord.File(buffer, filename="answer.md")
            msg = "⚠️ Answer was too long, I've sent it as a markdown file."
            if deferred:
                await ctx.followup.send(content=msg, file=file)
            else:
                await ctx.respond(content=msg, file=file)
    except Exception:
        logger.exception("Unhandled exception in project_query")
        fallback = "❌ An unexpected error occurred. Check logs."
        if deferred:
            await ctx.followup.send(fallback, ephemeral=True)
        else:
            await ctx.respond(fallback, ephemeral=True)

@bot.event
async def on_ready():
    print(f"Project Bot logged in as {bot.user}")

if TOKEN:
    bot.run(TOKEN)
else:
    print("ERROR: PROJECT_TOKEN not set in environment")
