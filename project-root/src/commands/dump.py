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

# Import your summarization module
from summarization import get_summarizer

# Load environment
load_dotenv()
TOKEN = env.get('DUMP_TOKEN')
DATA_DIR = Path(env.get('STORAGE_DIR', './data/files'))
DB_FILE = env.get('DB_FILE', './dumpbot.db')
LLAMA_MODEL_PATH = env.get('LLAMA_MODEL_PATH', '/path/to/llama-model.gguf')  # adjust as needed

# Ensure storage directory
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Discord intents
intents = discord.Intents.default()
intents.guilds = True
intents.messages = True
intents.message_content = True

# Bot instance
bot = discord.Bot(intents=intents)

# Globals for DB
_db = None
_db_initialized = False

# Initialize summarizer
summarizer = get_summarizer(LLAMA_MODEL_PATH)

# Helper: infer summary context from channel name
def determine_context(name: str) -> str:
    n = name.lower()
    if 'client' in n:
        return 'client'
    if 'project' in n:
        return 'project'
    if 'subproject' in n:
        return 'subproject'
    # treat any other (e.g. task) as subproject-level
    return 'subproject'

async def get_db():
    global _db, _db_initialized
    if _db is None:
        _db = await aiosqlite.connect(DB_FILE)
        _db.row_factory = aiosqlite.Row
    if not _db_initialized:
        # Core schema
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
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            resp.raise_for_status()
            return await resp.read()

async def ensure_channel_context(channel_id: str, channel_name: str):
    db = await get_db()
    cur = await db.execute(
        'SELECT document_id, ancestors, path FROM channels WHERE channel_id = ?',
        (channel_id,)
    )
    row = await cur.fetchone()
    if row:
        return {
            'parent_id': row['document_id'],
            'ancestors': row['ancestors'].split(',') if row['ancestors'] else [],
            'base_path': row['path']
        }
    # first dump: create root
    doc_id = str(uuid.uuid4())
    slug = slugify(channel_name, lowercase=True, max_length=100)
    now = datetime.utcnow().isoformat()
    await db.execute(
        'INSERT INTO documents (id, original_name, slug, parent_id, ancestors, path, uploaded_at) VALUES (?,?,?,?,?,?,?)',
        (doc_id, channel_name, slug, None, '', doc_id, now)
    )
    await db.execute(
        'INSERT INTO channels (channel_id, document_id, ancestors, path, created_at) VALUES (?,?,?,?,?)',
        (channel_id, doc_id, '', doc_id, now)
    )
    await db.commit()
    return {'parent_id': doc_id, 'ancestors': [], 'base_path': doc_id}

async def propagate_to_ancestors(doc_id: str):
    db = await get_db()
    row = await db.execute_fetchone(
        "SELECT parent_id, ancestors FROM documents WHERE id = ?", (doc_id,)
    )
    if not row:
        return
    parent = row['parent_id']
    ancestor_ids = row['ancestors'].split(',') if row['ancestors'] else []
    if parent:
        ancestor_ids.append(parent)

    for anc in ancestor_ids:
        children = await db.execute_fetchall(
            'SELECT blob_key FROM documents WHERE parent_id = ?', (anc,)
        )
        texts = []
        for c in children:
            try:
                texts.append((DATA_DIR / c['blob_key']).read_text(encoding='utf-8', errors='ignore'))
            except:
                pass
        idx_file = DATA_DIR / anc / f"{anc}-index.txt"
        idx_file.parent.mkdir(parents=True, exist_ok=True)
        idx_file.write_text("\n\n".join(texts), encoding='utf-8')

async def upload_and_save(buffer: bytes, original_name: str, content_type: str, size: int, channel_id: str, channel_name: str):
    ctx = await ensure_channel_context(channel_id, channel_name)
    parent_id = ctx['parent_id']
    ancestors = ctx['ancestors']
    base = ctx['base_path']

    # save raw file
    doc_id = str(uuid.uuid4())
    slug = slugify(original_name, lowercase=True, max_length=100)
    ext = original_name.rsplit('.', 1)[-1] if '.' in original_name else ''
    rel = Path(parent_id) / f"{doc_id}-{slug}{'.'+ext if ext else ''}"
    full = DATA_DIR / rel
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_bytes(buffer)

    # persist metadata
    new_ancs = ancestors + ([parent_id] if parent_id else [])
    db = await get_db()
    now = datetime.utcnow().isoformat()
    await db.execute(
        '''INSERT INTO documents (id, original_name, slug, blob_key, parent_id, ancestors, path, content_type, file_size, uploaded_at)
           VALUES (?,?,?,?,?,?,?,?,?,?)''',
        (doc_id, original_name, slug, str(rel), parent_id, ','.join(new_ancs), f"{base}/{doc_id}", content_type, size, now)
    )
    await db.commit()

    # update raw-text indexes
    await propagate_to_ancestors(doc_id)

    # read content for summarization
    try:
        raw = full.read_text(encoding='utf-8', errors='ignore')
    except:
        raw = ''
    # summarize for this level
    lvl = determine_context(channel_name)
    try:
        summ = summarizer.summarize(raw, context=lvl)
        sum_file = DATA_DIR / parent_id / f"{parent_id}-summary.txt"
        sum_file.parent.mkdir(parents=True, exist_ok=True)
        with open(sum_file, 'a', encoding='utf-8') as f:
            f.write(f"\n\nSummary for {doc_id} ({lvl}):\n{summ}")
    except Exception as e:
        print(f"Summarization error for {doc_id}: {e}")

    # if this is a 'task' doc, propagate its summary upward
    if 'task' in channel_name.lower():
        # include parent and all ancestors
        cascade = ancestors + ([parent_id] if parent_id else [])
        for anc in cascade:
            # fetch original_name to infer level
            row = await db.execute_fetchone("SELECT original_name FROM documents WHERE id = ?", (anc,))
            ctx_lvl = determine_context(row['original_name']) if row else 'project'
            try:
                anc_summ = summarizer.summarize(raw, context=ctx_lvl)
                anc_file = DATA_DIR / anc / f"{anc}-summary.txt"
                with open(anc_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n\nTask {doc_id} summary for {ctx_lvl}:\n{anc_summ}")
            except Exception as e:
                print(f"Ancestor summarization error {anc}: {e}")

    return {'id': doc_id, 'path': f"{base}/{doc_id}"}

@bot.slash_command(name='dump', description='Upload a file or text dump')
async def dump(ctx: discord.ApplicationContext, file: discord.Attachment = None, text: str = None):
    try:
        if not file and not text:
            return await ctx.respond('Provide a file or text.', ephemeral=True)
        if file and text:
            return await ctx.respond('Provide only one: file or text.', ephemeral=True)
        await ctx.defer()
        if file:
            buffer = await fetch_attachment(file.url)
            orig = file.filename
            ctype = file.content_type or 'application/octet-stream'
        else:
            ts = int(datetime.utcnow().timestamp())
            orig = f'pasted-text-{ts}.txt'
            buffer = text.encode('utf-8')
            ctype = 'text/plain'
        res = await upload_and_save(buffer, orig, ctype, len(buffer), str(ctx.channel_id), ctx.channel.name)
        embed = discord.Embed(title='Document Stored', color=0x2ecc71, timestamp=datetime.utcnow())
        embed.add_field(name='Name', value=orig)
        embed.add_field(name='Doc ID', value=res['id'])
        embed.add_field(name='Path', value=res['path'], inline=False)
        await ctx.followup.send(embed=embed)
    except Exception:
        traceback.print_exc()
        await ctx.followup.send('Error uploading document.', ephemeral=True)

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user}')

bot.run(TOKEN)
