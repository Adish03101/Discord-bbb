# project_bot.py
import os
from os import environ as env
from pathlib import Path

import aiosqlite
import discord
from discord import Option

# Adjust import path as needed
from rag import ingest_document, rag_generate

# --- Config ---
DB_FILE = env.get("DB_FILE", "./dumpbot.db")

# Discord bot setup
intents = discord.Intents.default()
intents.message_content = True
bot = discord.Bot(intents=intents)


async def get_project_root(channel_id: str) -> str | None:
    """
    Fetch the UUID of the 'project' root for this Discord channel from your SQL metadata.
    Assumes you tagged channel roots in `channels` table when dumping.
    """
    db = await aiosqlite.connect(DB_FILE)
    db.row_factory = aiosqlite.Row
    row = await db.execute_fetchone(
        "SELECT document_id FROM channels WHERE channel_id = ?", (channel_id,)
    )
    await db.close()
    return row["document_id"] if row else None


@bot.slash_command(
    name="project_query",
    description="Ask a question against this project's documents"
)
async def project_query(
    ctx: discord.ApplicationContext,
    query: Option(str, "Your question about this project")
):
    await ctx.defer()
    # 1) Find the project root ID from your metadata
    root_id = await get_project_root(str(ctx.channel_id))
    if not root_id:
        return await ctx.respond(
            "🛑 This channel isn't registered as a project dump channel.",
            ephemeral=True
        )

    # 2) Use your RAG pipeline
    answer = rag_generate(root_id, query, top_k=4)

    # 3) Return the answer
    embed = discord.Embed(title="Project RAG Answer", color=0x4B9CD3)
    embed.add_field(name="Question", value=query, inline=False)
    embed.add_field(name="Answer", value=answer, inline=False)
    await ctx.respond(embed=embed)


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")



TOKEN = env.get("DUMP_TOKEN")
bot.run(TOKEN)
