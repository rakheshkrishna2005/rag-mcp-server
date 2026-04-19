#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from bs4 import BeautifulSoup
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession
from mcp.types import SamplingMessage, TextContent
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
RAG_DIR = BASE_DIR / ".rag"
INDEX_PATH = RAG_DIR / "index.faiss"
META_PATH = RAG_DIR / "metadata.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".html", ".htm", ".pdf"}


@dataclass
class DocumentRecord:
    id: str
    source_path: str
    title: str
    mime_type: str
    text: str
    created_at: str


@dataclass
class ChunkRecord:
    id: str
    document_id: str
    source_path: str
    chunk_index: int
    text: str
    start: int
    end: int
    embedding_id: int | None = None


class RagStore:
    def __init__(self) -> None:
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.documents: list[DocumentRecord] = []
        self.chunks: list[ChunkRecord] = []
        self.index = faiss.IndexFlatIP(self.model.get_sentence_embedding_dimension())
        self.dimension = self.model.get_sentence_embedding_dimension()
        self._load_state()

    def _load_state(self) -> None:
        if META_PATH.exists():
            payload = json.loads(META_PATH.read_text(encoding="utf-8"))
            self.documents = [DocumentRecord(**item) for item in payload.get("documents", [])]
            self.chunks = [ChunkRecord(**item) for item in payload.get("chunks", [])]

        if INDEX_PATH.exists():
            self.index = faiss.read_index(str(INDEX_PATH))
            self.dimension = self.index.d

    def save(self) -> None:
        RAG_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(INDEX_PATH))
        META_PATH.write_text(
            json.dumps(
                {
                    "documents": [asdict(document) for document in self.documents],
                    "chunks": [asdict(chunk) for chunk in self.chunks],
                },
                indent=2,
                ensure_ascii=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def summary(self) -> dict[str, Any]:
        embedded = sum(1 for chunk in self.chunks if chunk.embedding_id is not None)
        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "embedded_chunks": embedded,
            "index_size": int(self.index.ntotal),
            "dimension": self.dimension,
            "embedding_model": EMBEDDING_MODEL,
            "top_k_default": 5,
            "chunk_size_default": 1200,
            "chunk_overlap_default": 200,
            "index_path": str(INDEX_PATH),
        }

    def load_documents(self, paths: list[str], replace_existing: bool = True) -> dict[str, Any]:
        discovered: list[Path] = []
        skipped: list[str] = []

        for raw_path in paths:
            path = Path(raw_path).expanduser().resolve()
            if not path.exists():
                skipped.append(str(path))
                continue

            if path.is_dir():
                for child in path.rglob("*"):
                    if child.is_file() and child.suffix.lower() in SUPPORTED_EXTENSIONS:
                        discovered.append(child)
                continue

            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                discovered.append(path)
            else:
                skipped.append(str(path))

        loaded: list[DocumentRecord] = []
        for path in discovered:
            document = self._read_document(path)
            if document is None:
                skipped.append(str(path))
                continue

            if replace_existing:
                self._remove_document_by_source_path(str(path))

            self.documents.append(document)
            loaded.append(document)

        self.save()
        return {
            "loaded": len(loaded),
            "skipped": skipped,
            "documents": [self._document_to_dict(document) for document in loaded],
        }

    def split_documents(
        self,
        document_ids: list[str] | None = None,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        replace_existing: bool = True,
    ) -> dict[str, Any]:
        selected_ids = set(document_ids or [])
        selected = [document for document in self.documents if not selected_ids or document.id in selected_ids]

        if replace_existing and selected:
            selected_paths = {document.source_path for document in selected}
            self.chunks = [chunk for chunk in self.chunks if chunk.source_path not in selected_paths]
            self._rebuild_index_from_chunks()

        new_chunks: list[ChunkRecord] = []
        for document in selected:
            for index, chunk in enumerate(self._chunk_text(document.text, chunk_size, chunk_overlap)):
                new_chunks.append(
                    ChunkRecord(
                        id=f"chunk-{len(self.chunks) + len(new_chunks) + 1}",
                        document_id=document.id,
                        source_path=document.source_path,
                        chunk_index=index,
                        text=chunk,
                        start=0,
                        end=0,
                    )
                )

        self.chunks.extend(new_chunks)
        self.save()
        return {"created": len(new_chunks), "chunks": [self._chunk_to_dict(chunk) for chunk in new_chunks]}

    def create_embeddings(self, chunk_ids: list[str] | None = None) -> dict[str, Any]:
        selected = [chunk for chunk in self.chunks if chunk_ids is None or chunk.id in set(chunk_ids)]
        if not selected:
            return {"embedded": 0, "model": EMBEDDING_MODEL}

        texts = [chunk.text for chunk in selected]
        vectors = self.model.encode(texts, normalize_embeddings=True)
        vectors = np.asarray(vectors, dtype="float32")

        for chunk in self.chunks:
            chunk.embedding_id = None

        self.index.reset()
        self.index.add(vectors)

        chunk_iter = iter(range(len(selected)))
        for chunk in selected:
            chunk.embedding_id = next(chunk_iter)

        self.save()
        return {"embedded": len(selected), "model": EMBEDDING_MODEL, "dimension": self.dimension}

    def store_vectors(self) -> dict[str, Any]:
        self.save()
        return {"saved": True, **self.summary()}

    def embed_query(self, query: str, include_vector: bool = False) -> dict[str, Any]:
        vector = np.asarray(self.model.encode([query], normalize_embeddings=True), dtype="float32")[0]
        result: dict[str, Any] = {"query": query, "dimensions": int(vector.shape[0]), "preview": vector[:12].tolist()}
        if include_vector:
            result["vector"] = vector.tolist()
        return result

    def query_retrieval(
        self,
        query: str,
        top_k: int = 5,
        source_paths: list[str] | None = None,
        document_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        source_path_filter = self._resolve_source_paths(source_paths)
        document_id_filter = set(document_ids or [])
        candidates = self._select_chunks(source_path_filter, document_id_filter)

        if not candidates:
            return {
                "query": query,
                "top_k": top_k,
                "filters": {
                    "source_paths": source_path_filter,
                    "document_ids": sorted(document_id_filter),
                },
                "matches": [],
                "top_chunks": [],
            }

        query_vector = np.asarray(self.model.encode([query], normalize_embeddings=True), dtype="float32")[0]

        if not source_path_filter and not document_id_filter and self.index.ntotal > 0:
            scores, ids = self.index.search(np.asarray([query_vector], dtype="float32"), min(top_k, self.index.ntotal))
            matches = self._matches_from_embedding_ids(scores[0], ids[0])
        else:
            candidate_vectors = np.asarray(
                self.model.encode([chunk.text for chunk in candidates], normalize_embeddings=True),
                dtype="float32",
            )
            scores = candidate_vectors @ query_vector
            ranked_indices = np.argsort(-scores)[: min(top_k, len(candidates))]
            matches = [self._chunk_match(candidates[index], float(scores[index])) for index in ranked_indices]

        return {
            "query": query,
            "top_k": top_k,
            "filters": {
                "source_paths": source_path_filter,
                "document_ids": sorted(document_id_filter),
            },
            "matches": matches,
            "top_chunks": matches,
        }

    def build_prompt(
        self,
        query: str,
        top_k: int = 5,
        source_paths: list[str] | None = None,
        document_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        retrieval = self.query_retrieval(query, top_k, source_paths, document_ids)
        matches = retrieval["matches"]
        context = "\n\n---\n\n".join(
            f"[{index + 1}] {match['source_path']} :: chunk {match['chunk_index']}\n{match['text']}"
            for index, match in enumerate(matches)
        ) or "No matching context found."

        prompt = "\n".join(
            [
                "Answer only from the retrieved context.",
                "If the context is insufficient, say that explicitly.",
                "Cite the source path and chunk index when possible.",
                "",
                f"User question: {query}",
                "",
                "Retrieved context:",
                context,
            ]
        )

        return {
            "query": query,
            "top_k": top_k,
            "filters": retrieval["filters"],
            "prompt": prompt,
            "context": context,
            "matches": matches,
            "top_chunks": matches,
        }

    def _remove_document_by_source_path(self, source_path: str) -> None:
        removed_ids = {document.id for document in self.documents if document.source_path == source_path}
        self.documents = [document for document in self.documents if document.source_path != source_path]
        self.chunks = [chunk for chunk in self.chunks if chunk.document_id not in removed_ids]
        self._rebuild_index_from_chunks()

    def _resolve_source_paths(self, source_paths: list[str] | None) -> list[str]:
        if not source_paths:
            return []

        resolved: list[str] = []
        for raw_path in source_paths:
            resolved.append(str(Path(raw_path).expanduser().resolve()))
        return resolved

    def _select_chunks(
        self,
        source_paths: list[str] | None = None,
        document_ids: set[str] | None = None,
    ) -> list[ChunkRecord]:
        selected: list[ChunkRecord] = []
        source_path_filter = set(source_paths or [])
        document_id_filter = document_ids or set()

        for chunk in self.chunks:
            if chunk.embedding_id is None:
                continue
            if source_path_filter and chunk.source_path not in source_path_filter:
                continue
            if document_id_filter and chunk.document_id not in document_id_filter:
                continue
            selected.append(chunk)

        return selected

    def _matches_from_embedding_ids(self, scores: np.ndarray, ids: np.ndarray) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        for score, embedding_id in zip(scores, ids):
            chunk = self._chunk_for_embedding_id(int(embedding_id))
            if chunk is None:
                continue
            matches.append(self._chunk_match(chunk, float(score)))
        return matches

    def _chunk_for_embedding_id(self, embedding_id: int) -> ChunkRecord | None:
        for chunk in self.chunks:
            if chunk.embedding_id == embedding_id:
                return chunk
        return None

    @staticmethod
    def _chunk_match(chunk: ChunkRecord, score: float) -> dict[str, Any]:
        return {
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "source_path": chunk.source_path,
            "chunk_index": chunk.chunk_index,
            "score": score,
            "text": chunk.text,
        }

    def _rebuild_index_from_chunks(self) -> None:
        self.index.reset()
        embedded_chunks = [chunk for chunk in self.chunks if chunk.embedding_id is not None]
        if not embedded_chunks:
            return

        vectors = self.model.encode([chunk.text for chunk in embedded_chunks], normalize_embeddings=True)
        vectors = np.asarray(vectors, dtype="float32")
        self.index.add(vectors)
        for idx, chunk in enumerate(embedded_chunks):
            chunk.embedding_id = idx

    def _read_document(self, path: Path) -> DocumentRecord | None:
        try:
            if path.suffix.lower() == ".pdf":
                reader = PdfReader(str(path))
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
                title = (reader.metadata.title if reader.metadata and reader.metadata.title else path.name)
                mime_type = "application/pdf"
            elif path.suffix.lower() in {".html", ".htm"}:
                html = path.read_text(encoding="utf-8", errors="ignore")
                soup = BeautifulSoup(html, "html.parser")
                title = soup.title.text.strip() if soup.title and soup.title.text else path.name
                text = soup.get_text("\n")
                mime_type = "text/html"
            else:
                text = path.read_text(encoding="utf-8", errors="ignore")
                title = path.name
                mime_type = "text/plain"

            return DocumentRecord(
                id=f"doc-{len(self.documents) + 1}",
                source_path=str(path),
                title=title,
                mime_type=mime_type,
                text=self._normalize_text(text),
                created_at="",
            )
        except Exception:
            return None

    def _chunk_text(self, text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
        text = self._normalize_text(text)
        if not text:
            return []

        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                window = text[start:end]
                cut = max(window.rfind("\n\n"), window.rfind("\n"), window.rfind(" "))
                if cut > max(0, len(window) - 300):
                    end = start + cut

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            if end >= len(text):
                break
            start = max(0, end - chunk_overlap)

        return chunks

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", text.replace("\r\n", "\n")).strip()

    @staticmethod
    def _document_to_dict(document: DocumentRecord) -> dict[str, Any]:
        return asdict(document)

    @staticmethod
    def _chunk_to_dict(chunk: ChunkRecord) -> dict[str, Any]:
        return asdict(chunk)


store = RagStore()
mcp = FastMCP("copilot-rag-mcp")


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=True)


@mcp.tool()
def load_documents(paths: list[str], replace_existing: bool = True) -> str:
    """Collect PDF, TXT, HTML, and Markdown documents from files or directories."""
    return _json(store.load_documents(paths, replace_existing))


@mcp.tool()
def split_documents(
    document_ids: list[str] | None = None,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    replace_existing: bool = True,
) -> str:
    """Break loaded documents into smaller overlapping chunks."""
    return _json(store.split_documents(document_ids, chunk_size, chunk_overlap, replace_existing))


@mcp.tool()
def create_embeddings(chunk_ids: list[str] | None = None) -> str:
    """Convert each chunk into a vector using a local embedding model."""
    return _json(store.create_embeddings(chunk_ids))


@mcp.tool()
def store_vectors() -> str:
    """Persist vectors and original text to the FAISS-backed local store."""
    return _json(store.store_vectors())


@mcp.tool()
def embed_query(query: str, include_vector: bool = False) -> str:
    """Convert a query into a vector using the same embedding model as document chunks."""
    return _json(store.embed_query(query, include_vector))


@mcp.tool()
def query_retrieval(
    query: str,
    top_k: int = 5,
    source_paths: list[str] | None = None,
    document_ids: list[str] | None = None,
) -> str:
    """Perform similarity search over stored vectors and return the top matching chunks."""
    return _json(store.query_retrieval(query, top_k, source_paths, document_ids))


@mcp.tool()
def build_prompt(
    query: str,
    top_k: int = 5,
    source_paths: list[str] | None = None,
    document_ids: list[str] | None = None,
) -> str:
    """Combine retrieved context with the user question into an answer-ready prompt."""
    return _json(store.build_prompt(query, top_k, source_paths, document_ids))


@mcp.tool()
async def generate_answer(
    query: str,
    top_k: int = 5,
    max_tokens: int = 900,
    source_paths: list[str] | None = None,
    document_ids: list[str] | None = None,
    ctx: Context[ServerSession, None] = None,
) -> str:
    """Retrieve context, build a grounded prompt, and ask the connected Copilot LLM to answer."""
    built = store.build_prompt(query, top_k, source_paths, document_ids)
    sampling_result = await ctx.session.create_message(
        messages=[SamplingMessage(role="user", content=TextContent(type="text", text=built["prompt"]))],
        max_tokens=max_tokens,
    )

    if isinstance(sampling_result.content, TextContent):
        answer = sampling_result.content.text
    else:
        answer = str(sampling_result.content)

    return _json({
        "query": query,
        "top_k": top_k,
        "filters": built["filters"],
        "matches": built["matches"],
        "top_chunks": built["top_chunks"],
        "answer": answer,
        "model": sampling_result.model,
        "stop_reason": getattr(sampling_result, "stop_reason", None),
    })


@mcp.tool()
def rag_status() -> str:
    """Report the number of loaded documents, chunks, and embedded vectors."""
    return _json(store.summary())


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()