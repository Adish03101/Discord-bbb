import asyncio
from pg_db import get_pool, upsert_document, get_document_bytes

async def main():
    pool = await get_pool()
    print("Connected.")

    # Dummy upload
    name = "example.txt"
    slug = "example"
    content = b"hello postgres stored blob"
    res = await upsert_document(
        pool,
        original_name=name,
        slug=slug,
        parent_id=None,
        content_type="text/plain",
        file_bytes=content,
        summary="test summary",
        channel_id="test-channel",
    )
    print("Upsert result:", res)

    doc_id = res["id"]
    retrieved = await get_document_bytes(pool, doc_id)
    print("Retrieved bytes:", retrieved)
    assert retrieved == content

asyncio.run(main())
