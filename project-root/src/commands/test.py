import os
import uuid
import traceback
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import discord
from discord import TextChannel
import aiohttp
import logging
from dotenv import load_dotenv
from slugify import slugify
import sys


# project-root/src should already be on sys.path; adjust if needed
sys.path.append("/home/black/Backup/Discord-bbb/project-root/src")

from llm_actions.summarization import get_summarizer
from core.pg_db import get_pool, upsert_document  # PostgreSQL helpers

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('discord_bot_debug.log'),
        logging.StreamHandler()  # This will also print to console
    ]
)
logger = logging.getLogger(__name__)

# ─── Config & Environment ───────────────────────────────────────────────────────
load_dotenv()
TOKEN = os.getenv("DUMP_TOKEN")
DATA_DIR = Path(os.getenv("STORAGE_DIR", "./data/files"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Summarizer (AI logic)
summarizer = get_summarizer()

# ─── Discord Bot Setup ─────────────────────────────────────────────────────────
intents = discord.Intents.default()
intents.guilds = True
intents.messages = True
intents.message_content = True  # Add this for message content access
bot = discord.Bot(intents=intents)

# ─── Hierarchy Configuration ────────────────────────────────────────────────────
HIERARCHY = ["project", "subproject"]
LEAF_THREAD_LEVEL = "task"

# ─── Database Helper ────────────────────────────────────────────────────────────
async def update_document_summary(pool, doc_id: str, summary: str):
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE documents
            SET summary = $1
            WHERE id = $2
            """,
            summary,
            doc_id,
        )

# ─── Helpers ────────────────────────────────────────────────────────────────────

def parse_channel_name(name: str):
    name = name.lower().strip()
    parts = name.split("-", 2)
    level = parts[0]

    if level == LEAF_THREAD_LEVEL:
        if len(parts) != 2:
            raise ValueError(f"Bad task name: {name}")
        return level, parts[1], None

    if level not in HIERARCHY:
        raise ValueError(f"Unknown level: {level}")

    if level == HIERARCHY[0]:  # project
        if len(parts) != 2:
            raise ValueError(f"Bad {level} format: {name}")
        return level, parts[1], None

    if len(parts) == 3:
        _, parent_slug, own_slug = parts
        return level, own_slug, parent_slug

    raise ValueError(f"Bad name format: {name}")


def build_base_path(level: str, own_slug: str, parent_base):
    if level == "project":
        return DATA_DIR / "projects" / own_slug
    if level == "subproject":
        return DATA_DIR / "projects" / parent_base / "subprojects" / own_slug
    if level == LEAF_THREAD_LEVEL:
        return DATA_DIR / "projects" / parent_base / "tasks" / own_slug
    return DATA_DIR / own_slug


async def ensure_channel_context(channel_id: str, channel_name: str, guild: discord.Guild):
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT document_id, ancestors, path FROM channels WHERE channel_id=$1",
            channel_id
        )
    if row:
        return {
            "parent_id": str(row["document_id"]),  # Ensure string
            "ancestors": [str(ancestor) for ancestor in (row["ancestors"] or [])],  # Ensure all strings
            "base_path": Path(row["path"]),
        }

    level, own_slug, parent_slug = parse_channel_name(channel_name)
    parent_id, ancestors, parent_base = None, [], None

    if level == LEAF_THREAD_LEVEL:
        thread = guild.get_channel(int(channel_id))
        parent_chan = thread.parent
        if not isinstance(parent_chan, TextChannel):
            raise RuntimeError(f"Thread parent not a TextChannel: {parent_chan}")
        parent_ctx = await ensure_channel_context(
            str(parent_chan.id), parent_chan.name, guild
        )
        parent_id = str(parent_ctx["parent_id"])  # Ensure string
        ancestors = parent_ctx["ancestors"] + [parent_id]
        parent_base = parent_ctx["base_path"].relative_to(DATA_DIR / "projects")

    elif parent_slug:
        idx = HIERARCHY.index(level)
        if idx == 0:
            parent_id, ancestors, parent_base = None, [], None
        else:
            parent_level = HIERARCHY[idx - 1]
            parent_name = f"{parent_level}-{parent_slug}"
            parent_chan = next(
                (ch for ch in guild.channels
                 if isinstance(ch, TextChannel) and ch.name == parent_name),
                None
            )
            if not parent_chan:
                available = [ch.name for ch in guild.channels if isinstance(ch, TextChannel)]
                raise RuntimeError(
                    f"Could not find parent channel {parent_name}. Available: {available}"
                )
            parent_ctx = await ensure_channel_context(
                str(parent_chan.id), parent_chan.name, guild
            )
            parent_id = str(parent_ctx["parent_id"])  # Ensure string
            ancestors = parent_ctx["ancestors"] + [parent_id]
            parent_base = parent_ctx["base_path"].relative_to(DATA_DIR / "projects")

    base_path = build_base_path(level, own_slug, parent_base)
    base_path.mkdir(parents=True, exist_ok=True)

    doc_res = await upsert_document(
        pool,
        original_name=channel_name,
        slug=f"{level}-{own_slug}",
        parent_id=parent_id,
        content_type=None,
        file_bytes=None,
        summary=None,
        channel_id=channel_id,
    )
    this_doc_id = str(doc_res["id"])  # Convert UUID to string

    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO channels (channel_id, document_id, ancestors, path) VALUES ($1,$2,$3,$4)",
            channel_id,
            this_doc_id,
            ancestors,
            str(base_path),
        )

    return {"parent_id": this_doc_id, "ancestors": ancestors, "base_path": base_path}


async def propagate_to_ancestors(doc_id: str):
    pool = await get_pool()
    async with pool.acquire() as conn:
        pass
async def check_db_connection():
    """Test database connection and show available tables"""
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            result = await conn.fetchval("SELECT 1")
            logger.info(f"✅ Database connection test: {result}")
            
            # Check if tables exist
            tables = await conn.fetch("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            table_names = [row['table_name'] for row in tables]
            logger.info(f"📋 Available tables: {table_names}")
            
            # Check documents table structure
            if 'documents' in table_names:
                columns = await conn.fetch("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = 'documents'
                """)
                logger.info(f"📄 Documents table columns: {[(col['column_name'], col['data_type']) for col in columns]}")
            
            # Check current document count
            doc_count = await conn.fetchval("SELECT COUNT(*) FROM documents")
            logger.info(f"📊 Current documents in DB: {doc_count}")
            
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise

async def test_direct_db_insert():
    """Test if we can directly insert into the database"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        try:
            # Test simple insert
            test_id = str(uuid.uuid4())
            await conn.execute("""
                INSERT INTO documents (id, original_name, slug, content_type, file_data, channel_id)
                VALUES ($1, $2, $3, $4, $5, $6)
            """, test_id, "test.txt", "test", "text/plain", b"test content", "123456789")
            
            # Verify insert
            result = await conn.fetchrow("SELECT * FROM documents WHERE id = $1", test_id)
            if result:
                logger.info(f"✅ Direct insert test successful: {result['original_name']}")
            else:
                logger.error("❌ Direct insert failed - no record found")
            
            # Clean up
            await conn.execute("DELETE FROM documents WHERE id = $1", test_id)
            logger.info("🧹 Test record cleaned up")
            
        except Exception as e:
            logger.error(f"❌ Direct database insert test failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")

async def upload_and_save_debug(
    buffer: bytes,
    name: str,
    ctype: str,
    size: int,
    chan_id: str,
    chan_name: str,
    guild: discord.Guild,
):
    logger.info(f"🚀 Starting upload_and_save for: {name}")
    logger.info(f"📍 Channel ID: {chan_id}, Channel Name: {chan_name}")
    logger.info(f"📦 Buffer size: {len(buffer)} bytes, Content type: {ctype}")
    
    try:
        # Step 1: Get channel context
        logger.info("📁 Getting channel context...")
        ctx = await ensure_channel_context(chan_id, chan_name, guild)
        logger.info(f"✅ Channel context obtained: {ctx}")
        
        parent_id = ctx["parent_id"]
        base_path = ctx["base_path"]
        
        logger.info(f"👥 Parent ID: {parent_id}")
        logger.info(f"📂 Base path: {base_path}")
        
        # Step 2: Generate file info
        doc_id = str(uuid.uuid4())
        slug = slugify(name)
        ext = Path(name).suffix
        blob = f"{doc_id}-{slug}{ext}"
        full_path = base_path / blob
        
        logger.info(f"🆔 Generated doc_id: {doc_id}")
        logger.info(f"🏷️ Slug: {slug}")
        logger.info(f"💾 Full path: {full_path}")
        
        # Step 3: Save to disk (if enabled)
        if os.getenv("DISABLE_DISK_CACHE") != "1":
            full_path.write_bytes(buffer)
            logger.info(f"✅ File written to disk: {full_path}")
        else:
            logger.info("⚠️ Disk cache disabled - skipping file write")

        # Step 4: Database operations
        logger.info("🗄️ Getting database pool...")
        pool = await get_pool()
        new_doc_id = None
        
        try:
            logger.info("💾 Attempting to upsert document...")
            logger.info(f"Parameters: name={name}, slug={slug}, parent_id={parent_id}, content_type={ctype}, channel_id={chan_id}")
            
            res = await upsert_document(
                pool,
                original_name=name,
                slug=slug,
                parent_id=parent_id,
                content_type=ctype,
                file_bytes=buffer,
                summary=None,
                channel_id=chan_id,
            )
            
            new_doc_id = str(res["id"])
            logger.info(f"✅ Document upserted successfully. New doc ID: {new_doc_id}")
            logger.info(f"📋 Upsert result: {res}")
            
            # Step 5: Verify the document was actually inserted
            async with pool.acquire() as conn:
                verify = await conn.fetchrow(
                    "SELECT id, original_name, parent_id, channel_id FROM documents WHERE id = $1",
                    new_doc_id
                )
                if verify:
                    logger.info(f"✅ Verification successful: {dict(verify)}")
                else:
                    logger.error("❌ Document not found after upsert!")
                    
                # Check total document count
                total_docs = await conn.fetchval("SELECT COUNT(*) FROM documents")
                logger.info(f"📊 Total documents in DB after upsert: {total_docs}")
                    
        except Exception as e:
            logger.error(f"❌ Database upsert failed: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise  # Re-raise to see the full error in Discord
        
        # Step 6: Handle summarization (with error handling)
        if new_doc_id:
            try:
                await propagate_to_ancestors(new_doc_id)
                logger.info("✅ Ancestor propagation completed")
            except Exception as e:
                logger.warning(f"⚠️ Ancestor propagation failed: {e}")
            
            try:
                text = full_path.read_text(errors="ignore") if full_path.exists() else buffer.decode(
                    "utf-8", errors="ignore"
                )
                summary = summarizer.summarize(text, context=parse_channel_name(chan_name)[0])
                
                # Save summary to file
                with open(base_path / "summaries.txt", "a") as sf:
                    sf.write(f"\n\n-- {doc_id} --\n{summary}")
                
                # Update database with summary
                if summary:
                    await update_document_summary(pool, new_doc_id, summary)
                    logger.info("✅ Summary generated and saved")
                
            except Exception as e:
                logger.warning(f"⚠️ Summarization failed: {e}")
                import traceback
                logger.warning(f"Summary traceback: {traceback.format_exc()}")
        
        return {"local_id": doc_id, "document_id": new_doc_id, "path": str(base_path)}
        
    except Exception as e:
        logger.error(f"💥 upload_and_save failed completely: {e}")
        import traceback
        logger.error(f"Complete failure traceback: {traceback.format_exc()}")
        raise

# ─── NEW APPROACH: Message-based file uploads ──────────────────────────────────
@bot.event
async def on_message(message):
    # Skip bot messages and DMs
    if message.author.bot or not message.guild:
        return
    
    # Only process messages that mention the bot and have attachments
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
            
            # Read attachment using Discord.py's built-in method
            buffer = await attachment.read()
            logger.info(f"📥 Successfully read {len(buffer)} bytes from attachment")
            
            # Upload and save using debug version
            result = await upload_and_save_debug(
                buffer,
                attachment.filename,
                attachment.content_type or "application/octet-stream",
                len(buffer),
                str(message.channel.id),
                message.channel.name,
                message.guild,
            )
            
            successful_uploads.append({
                'filename': attachment.filename,
                'doc_id': result["document_id"],
                'local_id': result["local_id"],
                'path': result["path"]
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
    
    # Send batch confirmation (same as your original code)
    if successful_uploads or failed_uploads:
        local_time = datetime.now(ZoneInfo("Asia/Kolkata")).strftime(
            "%Y-%m-%d %I:%M %p %Z"
        )
        
        embed = discord.Embed(
            title=f"Batch Upload Complete",
            color=0x2ecc71 if not failed_uploads else 0xe74c3c,
            timestamp=datetime.now(timezone.utc)
        )
        
        # Add successful uploads
        if successful_uploads:
            success_text = "\n".join([
                f"✅ `{item['filename']}` (ID: {item['doc_id']})"
                for item in successful_uploads
            ])
            embed.add_field(
                name=f"Successfully Stored ({len(successful_uploads)} files)",
                value=success_text[:1024],
                inline=False
            )
        
        # Add failed uploads
        if failed_uploads:
            fail_text = "\n".join([
                f"❌ `{item['filename']}`: {item['error'][:50]}..."
                for item in failed_uploads
            ])
            embed.add_field(
                name=f"Failed ({len(failed_uploads)} files)",
                value=fail_text[:1024],
                inline=False
            )
        
        if successful_uploads:
            embed.add_field(name="Folder", value=successful_uploads[0]["path"], inline=False)
        embed.add_field(name="Uploaded", value=local_time, inline=True)
        
        await message.reply(embed=embed)

# ─── IMPROVED SLASH COMMAND (text only to avoid ephemeral issues) ──────────────
@bot.slash_command(name="dump", description="Save text into the vault (for files, upload and mention the bot)")
async def dump(
    ctx: discord.ApplicationContext,
    text: str = None,
):
    if not text:
        return await ctx.respond(
            "📝 **How to use the dump feature:**\n\n"
            "**For text:** Use `/dump text:your_text_here`\n"
            "**For files:** Upload a file in a regular message and mention me (@bot_name)\n\n"
            "⚠️ Slash command file uploads have limitations due to Discord's ephemeral attachment system.",
            ephemeral=True
        )
    
    await ctx.defer()
    
    try:
        buf = text.encode()
        name = f"pasted-{int(datetime.now(timezone.utc).timestamp())}.txt"
        ctype = "text/plain"

        res = await upload_and_save_debug(
            buf, name, ctype, len(buf),
            str(ctx.channel_id), ctx.channel.name, ctx.guild
        )

        local_id = res["local_id"]
        document_id = res["document_id"]
        local_time = datetime.now(ZoneInfo("Asia/Kolkata")).strftime(
            "%Y-%m-%d %I:%M %p %Z"
        )

        embed = discord.Embed(
            title=f"Stored: {name}", color=0x2ecc71,
            timestamp=datetime.now(timezone.utc)
        )
        embed.add_field(name="Name", value=name, inline=True)
        embed.add_field(name="Document ID", value=str(document_id), inline=True)
        embed.add_field(name="Local ID", value=local_id, inline=True)
        embed.add_field(name="Folder", value=res["path"], inline=False)
        embed.add_field(name="Uploaded", value=local_time, inline=True)
        await ctx.followup.send(embed=embed)
        
    except Exception as e:
        print(f"[ERROR] Dump command failed: {e}")
        traceback.print_exc()
        await ctx.followup.send(
            f"❌ Failed to process text: {str(e)}",
            ephemeral=True
        )


@bot.slash_command(name="dump_status", description="Show the latest stored document metadata for this channel")
async def dump_status(ctx: discord.ApplicationContext):
    await ctx.defer()
    pool = await get_pool()
    channel_id = str(ctx.channel_id)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, original_name, slug, file_size,
                   uploaded_at, octet_length(file_data) AS bytes_length,
                   summary
            FROM documents
            WHERE channel_id=$1
            ORDER BY uploaded_at DESC
            LIMIT 1
            """,
            channel_id,
        )
    if not row:
        return await ctx.followup.send(
            f"No document found for this channel ({channel_id})."
        )

    local_time = row["uploaded_at"].astimezone(
        ZoneInfo("Asia/Kolkata")
    ).strftime("%Y-%m-%d %I:%M %p %Z")
    embed = discord.Embed(
        title="Latest Stored Document",
        color=0x3498db,
        timestamp=row["uploaded_at"],
    )
    embed.add_field(name="Name", value=row["original_name"], inline=True)
    embed.add_field(name="ID", value=str(row["id"]), inline=True)
    embed.add_field(name="Slug", value=row["slug"] or "-", inline=True)
    embed.add_field(name="Channel ID", value=channel_id, inline=True)
    embed.add_field(
        name="Size (bytes)", value=str(row["file_size"] or row["bytes_length"] or 0),
        inline=True
    )
    embed.add_field(name="Blob length", value=str(row["bytes_length"]), inline=True)
    embed.add_field(name="Uploaded", value=local_time, inline=True)
    embed.add_field(name="Has summary", value="yes" if row["summary"] else "no", inline=True)
    await ctx.followup.send(embed=embed)


@bot.event
async def on_ready():
    print(f"🤖 Logged in as {bot.user}")
    logger.info(f"Bot started: {bot.user}")
    
    # Test database connection and operations
    try:
        logger.info("🔍 Testing database connection...")
        await check_db_connection()
        
        logger.info("🧪 Testing direct database operations...")
        await test_direct_db_insert()
        
        print("✅ Database connection and operations working correctly")
        logger.info("✅ All database tests passed")
        
    except Exception as e:
        print(f"❌ Database issues detected: {e}")
        logger.error(f"Database startup test failed: {e}")
        print("⚠️ Bot may not function correctly - check logs for details")
    
    print("📁 To upload files: Upload a file and mention the bot in the same message")
    print("📝 To upload text: Use /dump command")


bot.run(TOKEN)