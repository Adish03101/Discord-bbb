import os
from os import environ as env
from dotenv import load_dotenv
import discord
import traceback
from datetime import datetime
import uuid
from pathlib import Path
from slugify import slugify
import asyncio
import aiosqlite
import aiohttp

# Load environment
load_dotenv()
TOKEN = env.get('DUMP_TOKEN')
DATA_DIR = Path(env.get('STORAGE_DIR', './data/files'))
DB_FILE = env.get('DB_FILE', './dumpbot.db')
LOG_FILE = 'dumpbot_debug.log'

# Ensure storage directory
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Discord intents
intents = discord.Intents.default()
intents.guilds = True
intents.messages = True
intents.message_content = True

# PyCord Bot instance
bot = discord.Bot(intents=intents)

# Global DB connection
_db: aiosqlite.Connection = None
_db_initialized = False

async def get_db():
    global _db, _db_initialized
    if _db is None:
        _db = await aiosqlite.connect(DB_FILE)
        _db.row_factory = aiosqlite.Row
    if not _db_initialized:
        # Create tables if they don't exist
        await _db.execute("""
            CREATE TABLE IF NOT EXISTS documents (
              id TEXT PRIMARY KEY,
              original_name TEXT NOT NULL,
              slug TEXT NOT NULL,
              blob_key TEXT UNIQUE,
              parent_id TEXT,
              ancestors TEXT NOT NULL,
              path TEXT NOT NULL,
              content_type TEXT,
              file_size INTEGER,
              uploaded_at TEXT NOT NULL
            );
        """)
        await _db.execute("""
            CREATE TABLE IF NOT EXISTS channels (
              channel_id TEXT PRIMARY KEY,
              document_id TEXT NOT NULL,
              ancestors TEXT NOT NULL,
              path TEXT NOT NULL,
              created_at TEXT NOT NULL
            );
        """)
        await _db.commit()
        _db_initialized = True
    return _db

async def fetch_attachment(url: str) -> bytes:
    """
    Fetches raw bytes of an attachment URL using aiohttp.
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            resp.raise_for_status()
            return await resp.read()

async def ensure_channel_context(channel_id: str, channel_name: str):
    db = await get_db()
    cursor = await db.execute(
        'SELECT document_id, ancestors, path FROM channels WHERE channel_id = ?',
        (channel_id,)
    )
    row = await cursor.fetchone()
    if row:
        return {
            'parent_id': row['document_id'],
            'ancestors': row['ancestors'].split(',') if row['ancestors'] else [],
            'base_path': row['path']
        }
    # Fallback: use file content classification to assign metadata
    # Here you could call an LLM to infer parent_id or tags based on channel_name
    doc_id = str(uuid.uuid4())
    slug = slugify(channel_name, lowercase=True, max_length=100)
    ancestors = []
    path = doc_id
    now = datetime.utcnow().isoformat()
    await db.execute(
        'INSERT INTO documents (id, original_name, slug, parent_id, ancestors, path, uploaded_at) VALUES (?,?,?,?,?,?,?)',
        (doc_id, channel_name, slug, None, '', path, now)
    )
    await db.execute(
        'INSERT INTO channels (channel_id, document_id, ancestors, path, created_at) VALUES (?,?,?,?,?)',
        (channel_id, doc_id, '', path, now)
    )
    await db.commit()
    return {'parent_id': doc_id, 'ancestors': ancestors, 'base_path': path}

async def upload_and_save(buffer: bytes, original_name: str, content_type: str, size: int, channel_id: str, channel_name: str):
    # Ensure or infer metadata context
    ctx = await ensure_channel_context(channel_id, channel_name)
    parent_id = ctx['parent_id']
    ancestors = ctx['ancestors']
    base_path = ctx['base_path']

    # Generate doc ID & slug
    doc_id = str(uuid.uuid4())
    slug = slugify(original_name, lowercase=True, max_length=100)
    ext = original_name.rsplit('.', 1)[-1] if '.' in original_name else ''
    rel_path = Path(parent_id) / f"{doc_id}-{slug}{'.'+ext if ext else ''}"
    full_path = DATA_DIR / rel_path
    full_path.parent.mkdir(parents=True, exist_ok=True)

    # Save file locally
    with open(full_path, 'wb') as f:
        f.write(buffer)

    # Build metadata
    new_ancestors = ancestors + [parent_id]
    path = f"{base_path}/{doc_id}"
    blob_key = str(rel_path)
    now = datetime.utcnow().isoformat()

    # Persist metadata
    db = await get_db()
    await db.execute(
        '''
        INSERT INTO documents (
          id, original_name, slug, blob_key,
          parent_id, ancestors, path,
          content_type, file_size, uploaded_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?)
        ''',
        (doc_id, original_name, slug, blob_key,
         parent_id, ','.join(new_ancestors), path,
         content_type, size, now)
    )
    await db.commit()
    return {'id': doc_id, 'path': path}

# --- /dump Command ---
@bot.slash_command(name='dump', description='Upload a file or text dump')
async def dump(ctx: discord.ApplicationContext, file: discord.Attachment = None, text: str = None):
    try:
        if not file and not text:
            return await ctx.respond('Provide a file or text.', ephemeral=True)
        if file and text:
            return await ctx.respond('Provide only one: file or text.', ephemeral=True)

        await ctx.defer()
        if file:
            # Download attachment bytes
            buffer = await fetch_attachment(file.url)
            original_name = file.filename
            content_type = file.content_type or 'application/octet-stream'
            size = len(buffer)
        else:
            ts = int(datetime.utcnow().timestamp())
            original_name = f'pasted-text-{ts}.txt'
            buffer = text.encode('utf-8')
            content_type = 'text/plain'
            size = len(buffer)

        result = await upload_and_save(buffer, original_name, content_type, size,
                                       str(ctx.channel_id), ctx.channel.name)

        # Reply
        embed = discord.Embed(title='Document Stored', color=0x2ecc71, timestamp=datetime.utcnow())
        embed.add_field(name='Name', value=original_name, inline=True)
        embed.add_field(name='Doc ID', value=result['id'], inline=True)
        embed.add_field(name='Path', value=result['path'], inline=False)
        await ctx.followup.send(embed=embed)
    except Exception:
        traceback.print_exc()
        await ctx.followup.send('Error uploading document.', ephemeral=True)

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user}')

# --- Run Bot ---
bot.run(TOKEN)