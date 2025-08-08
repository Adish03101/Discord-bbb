import os
import re
import time
import math
import logging
import asyncio
import pickle
import numpy as np  # FIXED: Import numpy at module level
from pathlib import Path
from typing import List, Dict, Optional

import asyncpg
import requests
from dotenv import load_dotenv
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

# ─── Setup ─────────────────────────────────────────────────────────────────────

nltk.download("punkt", quiet=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_faiss")

# Load env
env_path = Path(__file__).resolve().parent.parent / "commands" / ".env"
if not env_path.exists():
    env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# ─── Configuration ─────────────────────────────────────────────────────────────

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GEN_MODEL = os.getenv("GEN_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
CHAT_URL = f"https://generativelanguage.googleapis.com/v1/models/{GEN_MODEL}:generateContent"

# ADDED: Hugging Face configuration
HF_GEN_MODEL = os.getenv("HF_GEN_MODEL", "openai/gpt-oss-20b:fireworks-ai")
HF_API_URL = "https://router.huggingface.co/v1/chat/completions"

HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
HYBRID_SPARSE = os.getenv("HYBRID_SPARSE", "True").lower() in ("1", "true", "yes")
INITIAL_RETRIEVE_K = int(os.getenv("INITIAL_RETRIEVE_K", "512"))
TOP_K = int(os.getenv("TOP_K", "2"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "800"))

# Storage for per-root FAISS indexes and metadata
INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", "./faiss_indices"))
INDEX_DIR.mkdir(parents=True, exist_ok=True)

_faiss_indexes: Dict[str, faiss.Index] = {}
_metadata_store: Dict[str, Dict] = {} # root_id -> {"chunk_ids": [...], "texts": [...], "dim": int}

# ─── Embedding fallback (local) ─────────────────────────────────────────────────

_local_embedder: Optional[SentenceTransformer] = None

if not HF_TOKEN:
    try:
        _local_embedder = SentenceTransformer(EMBEDDING_MODEL)
    except Exception as e:
        logger.warning("Failed to load local embedder: %s", e)
        _local_embedder = None

def _local_embed_texts(texts: List[str]) -> List[List[float]]:
    if not _local_embedder:
        raise RuntimeError("Local embedder not available")
    embs = _local_embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embs.tolist()

def embed_texts(texts: List[str]) -> List[List[float]]:
    embeddings: List[List[float]] = []
    if HF_TOKEN:
        # Simple HF inference placeholder: you can replace with real HF API calls if needed
        for t in texts:
            # Fallback to local since dedicated HF embed endpoint may not exist
            if _local_embedder:
                emb = _local_embed_texts([t])[0]
            else:
                emb = [0.0] * 768
            embeddings.append(emb)
    else:
        embeddings = _local_embed_texts(texts)

    # normalize to unit length
    normalized: List[List[float]] = []
    for v in embeddings:
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        normalized.append([x / norm for x in v])
    return normalized

# ─── Cross-encoder reranker setup ───────────────────────────────────────────────

_cross_encoder: Optional[CrossEncoder] = None

def get_cross_encoder() -> Optional[CrossEncoder]:
    global _cross_encoder
    if _cross_encoder is None:
        try:
            _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as e:
            logger.warning("Failed to load CrossEncoder: %s", e)
            _cross_encoder = None
    return _cross_encoder

def rerank_with_crossencoder(query: str, passages: List[str], fused_scores: Optional[List[float]] = None) -> List[float]:
    ce = get_cross_encoder()
    if ce is None:
        return fused_scores if fused_scores is not None else [0.0] * len(passages)
    
    pairs = [[query, p] for p in passages]
    try:
        scores = ce.predict(pairs)
        if fused_scores and all(abs(s - scores[0]) < 1e-6 for s in scores):
            return fused_scores
        return scores.tolist()
    except Exception:
        logger.exception("CrossEncoder failed; using fused fallback.")
        return fused_scores if fused_scores is not None else [0.0] * len(passages)

# ─── Sparse index ─────────────────────────────────────────────────────────────

class SparseIndex:
    def __init__(self, texts: List[str]):
        tokenized = [word_tokenize(t.lower()) for t in texts]
        self.bm25 = BM25Okapi(tokenized)
        self.texts = texts

    def get_scores(self, query: str) -> List[float]:
        q_tok = word_tokenize(query.lower())
        return self.bm25.get_scores(q_tok)

sparse_store: Dict[str, SparseIndex] = {}

# ─── Chunking ─────────────────────────────────────────────────────────────────

def sentence_sliding_window(text: str, window_size: int = 5, overlap: int = 2) -> List[str]:
    sentences = sent_tokenize(text)
    if not sentences:
        return []

    chunks = []
    i = 0
    while i < len(sentences):
        window = sentences[i: i + window_size]
        chunks.append(" ".join(window))
        if i + window_size >= len(sentences):
            break
        i += window_size - overlap
    return chunks

# ─── FAISS helpers ────────────────────────────────────────────────────────────

def get_faiss_index(root_id: str, embedding_dim: int) -> faiss.Index:
    if root_id in _faiss_indexes:
        return _faiss_indexes[root_id]

    # try load from disk
    idx_file = INDEX_DIR / f"{root_id}.index"
    if idx_file.exists():
        index = faiss.read_index(str(idx_file))
        _faiss_indexes[root_id] = index
        return index

    # create new flat index with inner product (cosine after normalization)
    index = faiss.IndexFlatIP(embedding_dim)
    _faiss_indexes[root_id] = index
    return index

def persist_faiss_index(root_id: str):
    index = _faiss_indexes.get(root_id)
    if index is None:
        return
    faiss.write_index(index, str(INDEX_DIR / f"{root_id}.index"))

    # persist metadata
    meta = _metadata_store.get(root_id, {})
    with open(INDEX_DIR / f"{root_id}_meta.pkl", "wb") as f:
        pickle.dump(meta, f)

def load_metadata(root_id: str):
    meta_file = INDEX_DIR / f"{root_id}_meta.pkl"
    if meta_file.exists():
        try:
            with open(meta_file, "rb") as f:
                _metadata_store[root_id] = pickle.load(f)
            logger.info(f"✅ Loaded metadata for root_id={root_id}: {len(_metadata_store[root_id].get('texts', []))} chunks")
        except Exception as e:
            logger.error(f"❌ Failed to load metadata for {root_id}: {e}")
            # Clear corrupted metadata file
            meta_file.unlink(missing_ok=True)
    else:
        logger.info(f"⚠️ No metadata file found for root_id={root_id}")

def ingest_documents(root_id: str, texts: List[str]):
    if not texts:
        logger.warning("No texts to ingest for root_id=%s", root_id)
        return

    logger.info(f"🚀 Starting ingestion for root_id={root_id} with {len(texts)} texts")

    all_chunks: List[str] = []
    for i, t in enumerate(texts):
        if not t or not t.strip():
            logger.warning(f"Skipping empty text #{i}")
            continue
        chunks = sentence_sliding_window(t)
        logger.info(f"Text #{i}: {len(t)} chars -> {len(chunks)} chunks")
        all_chunks.extend(chunks)

    if not all_chunks:
        logger.warning("No chunks extracted for root_id=%s", root_id)
        return

    logger.info(f"📚 Total chunks to embed: {len(all_chunks)}")

    # embeddings
    try:
        embeddings = embed_texts(all_chunks)
        embedding_dim = len(embeddings[0]) if embeddings else 768
        logger.info(f"✅ Generated {len(embeddings)} embeddings of dimension {embedding_dim}")
    except Exception as e:
        logger.error(f"❌ Failed to generate embeddings: {e}")
        return

    # FIXED: Clear existing index before adding new vectors to maintain alignment
    try:
        # Create fresh index to avoid misalignment
        index = faiss.IndexFlatIP(embedding_dim)
        _faiss_indexes[root_id] = index
        
        # convert to numpy float32 array
        emb_array = np.array(embeddings, dtype="float32")
        index.add(emb_array) # assumes already normalized
        logger.info(f"✅ Added {len(embeddings)} vectors to FAISS index (total now: {index.ntotal})")
    except Exception as e:
        logger.error(f"❌ Failed to add to FAISS index: {e}")
        return

    # FIXED: Complete metadata storage
    chunk_ids = [f"{root_id}_{i}" for i in range(len(all_chunks))]
    _metadata_store[root_id] = {
        "chunk_ids": chunk_ids,
        "texts": all_chunks,
        "dim": embedding_dim,
    }

    if HYBRID_SPARSE:
        try:
            sparse_store[root_id] = SparseIndex(all_chunks)
            logger.info("✅ Created sparse BM25 index")
        except Exception as e:
            logger.error(f"❌ Failed to create sparse index: {e}")

    # persist
    try:
        persist_faiss_index(root_id)
        logger.info("✅ Persisted FAISS index and metadata to disk")
    except Exception as e:
        logger.error(f"❌ Failed to persist index: {e}")

    logger.info("✅ Ingested %d chunks for root_id=%s", len(all_chunks), root_id)

def retrieve_context(root_id: str, query: str, top_k: int = TOP_K) -> List[str]:
    # ensure metadata is loaded
    load_metadata(root_id)

    meta = _metadata_store.get(root_id)
    if not meta:
        logger.warning("No metadata for root_id=%s - this means index was never built", root_id)
        return []

    chunk_ids = meta.get("chunk_ids", [])
    texts = meta.get("texts", [])
    embedding_dim = meta.get("dim", 768)

    if not texts:
        logger.warning("No texts in metadata for root_id=%s", root_id)
        return []

    logger.info(f"Available chunks for search: {len(texts)}")

    # dense search
    try:
        q_emb = embed_texts([query])[0]
        q_vec = np.array([q_emb], dtype="float32")

        index = get_faiss_index(root_id, embedding_dim)
        if index.ntotal == 0:
            logger.warning("FAISS index is empty for root_id=%s", root_id)
            return []

        logger.info(f"FAISS index has {index.ntotal} vectors")

        # search top INITIAL_RETRIEVE_K (but don't exceed available documents)
        search_k = min(INITIAL_RETRIEVE_K, index.ntotal)
        D, I = index.search(q_vec, search_k) # inner product scores

        logger.info(f"Search returned {len(D[0])} results with scores: {D[0][:5]}")

        candidates = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(texts):
                continue
            candidates.append({
                "chunk_id": chunk_ids[idx] if idx < len(chunk_ids) else f"{root_id}_{idx}",
                "text": texts[idx],
                "score_dense": float(score),
            })

        if not candidates:
            logger.warning("No valid candidates found for root_id=%s", root_id)
            return []

        logger.info(f"Found {len(candidates)} valid candidates")

        # hybrid fusion
        if HYBRID_SPARSE and root_id in sparse_store:
            sparse_scores_raw = sparse_store[root_id].get_scores(query)
            sparse_scores = list(sparse_scores_raw) if sparse_scores_raw is not None else []

            max_sparse = max(sparse_scores) if sparse_scores else 1.0
            min_sparse = min(sparse_scores) if sparse_scores else 0.0

            dense_vals = [c["score_dense"] for c in candidates]
            max_dense = max(dense_vals) if dense_vals else 1.0
            min_dense = min(dense_vals) if dense_vals else 0.0

            for cand in candidates:
                # FIXED: Proper regex escaping
                m = re.match(rf"{re.escape(root_id)}_(\\d+)", cand.get("chunk_id", "") or "")
                if m:
                    idx = int(m.group(1))
                    raw_sparse = sparse_scores[idx] if 0 <= idx < len(sparse_scores) else 0.0
                    cand["score_sparse"] = (raw_sparse - min_sparse) / (max_sparse - min_sparse) if max_sparse > min_sparse else 0.0
                else:
                    cand["score_sparse"] = 0.0

                cand["score_dense_norm"] = (cand["score_dense"] - min_dense) / (max_dense - min_dense) if max_dense > min_dense else cand["score_dense"]
                cand["fused_score"] = (cand["score_dense_norm"] + cand["score_sparse"]) / 2.0
        else:
            for cand in candidates:
                cand["fused_score"] = cand["score_dense"]

        # rerank with cross-encoder
        rerank_pool = sorted(candidates, key=lambda x: x.get("fused_score", 0.0), reverse=True)[: max(top_k * 5, top_k)]
        texts_to_rerank = [c["text"] for c in rerank_pool]
        fused_scores = [c.get("fused_score", 0.0) for c in rerank_pool]

        cross_scores = rerank_with_crossencoder(query, texts_to_rerank, fused_scores=fused_scores)
        for i, cand in enumerate(rerank_pool):
            cand["cross_score"] = cross_scores[i] if i < len(cross_scores) else 0.0

        final = sorted(rerank_pool, key=lambda x: x.get("cross_score", 0.0), reverse=True)[: top_k]

        logger.info(f"✅ Retrieved {len(final)} chunks for root_id={root_id}")
        
        # FIXED: Add debugging for retrieved chunks
        for i, chunk in enumerate(final):
            logger.info(f"Chunk {i+1} (score: {chunk.get('cross_score', 0.0):.3f}): {chunk['text'][:100]}...")

        return [c["text"] for c in final]

    except Exception as e:
        logger.error(f"❌ retrieve_context failed for root_id={root_id}: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return []

# ADDED: Hugging Face API generator function
def call_hf_generator(context: str, query: str) -> str:
    if not HF_TOKEN:
        return "[Generator Error] No Hugging Face token configured."

    # FIXED: Add context validation and better debugging
    if not context or not context.strip():
        logger.warning("Empty context passed to generator")
        return "[RAG Error] No relevant context found in the documents."

    logger.info(f"Context length: {len(context)} characters")
    logger.info(f"Context preview: {context[:200]}...")

    payload = {
        "messages": [
            {
                "role": "user",
                "content": f"You are a helpful assistant. Answer the question based on the provided context. If the context doesn't contain relevant information, say so clearly.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
            }
        ],
        "model": HF_GEN_MODEL,
        "max_tokens": 500,
        "temperature": 0.3,
        "top_p": 0.9
    }

    try:
        response = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        # Handle HF API response format
        if "choices" not in result or not result["choices"]:
            logger.error(f"No choices in HF response: {result}")
            return "[Generator Error] Invalid response format from Hugging Face API."
        
        choice = result["choices"][0]
        
        # Check finish reason
        finish_reason = choice.get("finish_reason", "")
        if finish_reason == "length":
            logger.warning("Response was truncated due to max_tokens limit")
        
        # Get the message content
        if "message" not in choice:
            logger.error(f"No message in choice: {choice}")
            return "[Generator Error] No message in response."
        
        message = choice["message"]
        if "content" not in message:
            logger.error(f"No content in message: {message}")
            return "[Generator Error] No content in response message."
        
        answer = message["content"].strip()
        logger.info(f"HF Generator response: {answer}")
        return answer
        
    except requests.RequestException as e:
        logger.error("HF API request failed: %s", e)
        return "[Generator Error] Hugging Face API request failed."
    except KeyError as e:
        logger.error("Missing key in HF response: %s", e)
        logger.error(f"Full HF response: {result}")
        return f"[Generator Error] Missing key in HF response: {e}"
    except Exception as e:
        logger.error("HF Generator call failed: %s", e)
        return "[Generator Error] Unexpected error calling Hugging Face API."

# COMMENTED OUT: Original Gemini generator function
def call_generator(context: str, query: str) -> str:
    # Use Hugging Face API instead of Gemini
    return call_hf_generator(context, query)

    # COMMENTED OUT: Original Gemini code
    # if not GEMINI_API_KEY:
    #     return "[Generator Error] No Gemini key configured."

    # # FIXED: Add context validation and better debugging
    # if not context or not context.strip():
    #     logger.warning("Empty context passed to generator")
    #     return "[RAG Error] No relevant context found in the documents."

    # logger.info(f"Context length: {len(context)} characters")
    # logger.info(f"Context preview: {context[:200]}...")

    # payload = {
    #     "contents": [
    #         {"role": "user", "parts": [
    #             {"text": f"You are a helpful assistant. Answer the question based on the provided context. If the context doesn't contain relevant information, say so clearly.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"}
    #         ]}
    #     ],
    #     "generationConfig": {
    #         "temperature": 0.3,
    #         "maxOutputTokens": 300,  # FIXED: Increased from 300 to 500
    #         "topP": 0.9,
    #     }
    # }

    # try:
    #     response = requests.post(f"{CHAT_URL}?key={GEMINI_API_KEY}", json=payload, timeout=10)
    #     response.raise_for_status()
    #     result = response.json()
        
    #     # FIXED: Handle different response scenarios
    #     if "candidates" not in result or not result["candidates"]:
    #         logger.error(f"No candidates in response: {result}")
    #         return "[Generator Error] Invalid response format."
        
    #     candidate = result["candidates"][0]
    #     finish_reason = candidate.get("finishReason", "")
        
    #     # FIXED: Handle MAX_TOKENS case
    #     if finish_reason == "MAX_TOKENS":
    #         logger.error("Response was truncated due to MAX_TOKENS limit")
    #         return "[Generator Error] Response was truncated. Try asking a more specific question or check if the context is too long."
        
    #     # FIXED: Handle other finish reasons
    #     if finish_reason not in ["STOP", ""]:
    #         logger.error(f"Unexpected finish reason: {finish_reason}")
    #         return f"[Generator Error] Generation stopped due to: {finish_reason}"
        
    #     # FIXED: Check for content and parts
    #     if "content" not in candidate:
    #         logger.error(f"No content in candidate: {candidate}")
    #         return "[Generator Error] No content in response."
        
    #     content = candidate["content"]
        
    #     # FIXED: Handle missing parts (which happens with MAX_TOKENS)
    #     if "parts" not in content or not content["parts"]:
    #         logger.error(f"No parts in content: {content}")
    #         if finish_reason == "MAX_TOKENS":
    #             return "[Generator Error] Response was truncated due to token limit. Try a shorter context or more specific question."
    #         return "[Generator Error] No text parts in response."
        
    #     part = content["parts"][0]
    #     if "text" not in part:
    #         logger.error(f"No text in part: {part}")
    #         return "[Generator Error] No text in response part."
        
    #     answer = part["text"].strip()
    #     logger.info(f"Generator response: {answer}")
    #     return answer
        
    # except requests.RequestException as e:
    #     logger.error("HTTP request failed: %s", e)
    #     return "[Generator Error] HTTP request failed."
    # except KeyError as e:
    #     logger.error("Missing key in response: %s", e)
    #     logger.error(f"Full response: {result}")
    #     return f"[Generator Error] Missing key: {e}"
    # except Exception as e:
    #     logger.error("Generator call failed: %s", e)
    #     return "[Generator Error] Unexpected error."

def rag_generate(root_id: str, query: str, top_k: int = TOP_K) -> str:
    logger.info(f"🔍 Starting RAG generation for query: '{query}' with root_id: {root_id}")
    
    snippets = retrieve_context(root_id, query, top_k)
    if not snippets:
        logger.warning("No context snippets retrieved")
        return "[RAG Error] No relevant context found in the documents for your query."

    context = "\n---\n".join(snippets)
    logger.info(f"Combined context length: {len(context)} characters")
    
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS].rsplit("\n", 1)[0] + "\n..."
        logger.info(f"Context truncated to {len(context)} characters")

    return call_generator(context, query)

# ─── FIXED: Postgres fetch - only direct children, not recursive ─────────────────

async def fetch_texts_from_subproject(subproject_doc_id: str, database_url: str) -> List[str]:
    """FIXED: Fetch only documents directly under this subproject (no recursion)"""
    pool = await asyncpg.create_pool(database_url)
    try:
        async with pool.acquire() as conn:
            # FIXED: Only get documents where parent_id = subproject_doc_id
            # This prevents fetching documents from child projects/subprojects
            rows = await conn.fetch(
                """
                SELECT file_data FROM documents 
                WHERE parent_id::text = $1 
                ORDER BY uploaded_at DESC
                """,
                subproject_doc_id
            )
    finally:
        await pool.close()

    texts: List[str] = []
    for r in rows:
        if r["file_data"]:
            try:
                texts.append(r["file_data"].decode("utf-8", errors="ignore"))
            except Exception:
                continue

    logger.info(f"✅ Fetched {len(texts)} documents directly under subproject {subproject_doc_id}")
    return texts

# ─── DEPRECATED: Keep old function for backward compatibility ─────────────────

async def fetch_texts_from_channel(channel_id: str, database_url: str) -> List[str]:
    """DEPRECATED: Use fetch_texts_from_subproject instead"""
    logger.warning("fetch_texts_from_channel is deprecated, use fetch_texts_from_subproject")
    pool = await asyncpg.create_pool(database_url)
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow("SELECT document_id FROM channels WHERE channel_id=$1", channel_id)
            if not row:
                return []

            root_id = row["document_id"]
            # Use the new non-recursive approach
            rows = await conn.fetch(
                """
                SELECT file_data FROM documents 
                WHERE parent_id::text = $1 
                ORDER BY uploaded_at DESC
                """,
                str(root_id)
            )
    finally:
        await pool.close()

    texts: List[str] = []
    for r in rows:
        if r["file_data"]:
            try:
                texts.append(r["file_data"].decode("utf-8", errors="ignore"))
            except Exception:
                continue

    return texts

def answer_query(
    root_id: Optional[str],
    query: str,
    texts: Optional[List[str]] = None,
    database_url: Optional[str] = None,
    top_k: int = TOP_K,
) -> str:
    if not root_id:
        root_id = "default"

    logger.info(f"🎯 Answering query for root_id: {root_id}")

    if texts is None:
        if database_url:
            logger.info("Fetching texts from database...")
            texts = asyncio.run(fetch_texts_from_subproject(root_id, database_url))
        else:
            raise ValueError("Must supply either texts or database_url to answer_query.")

    logger.info(f"Available texts count: {len(texts) if texts else 0}")

    # FIXED: Only ingest if not already indexed
    load_metadata(root_id)
    if root_id not in _metadata_store or not _metadata_store[root_id].get("texts"):
        logger.info("No existing index found, creating new one...")
        ingest_documents(root_id, texts)
    else:
        logger.info(f"Using existing index with {len(_metadata_store[root_id].get('texts', []))} chunks")

    snippets = retrieve_context(root_id, query, top_k=top_k)
    if not snippets:
        return "[RAG Error] No relevant context found in the documents for your query."

    context = "\n---\n".join(snippets)
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS].rsplit("\n", 1)[0] + "\n..."

    return call_generator(context, query)

__all__ = [
    "ingest_documents",
    "retrieve_context", 
    "rag_generate",
    "list_indexed_ids",
    "fetch_texts_from_channel",
    "fetch_texts_from_subproject", # ADDED: New function
    "answer_query",
]