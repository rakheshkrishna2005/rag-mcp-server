# Copilot Instructions

Use the local Python `copilot-rag-mcp` server for retrieval-augmented work in this workspace.

Workflow expectations:

1. Ask the user to select one or more document paths before retrieval.
2. Use selected paths as `source_paths` for retrieval and answer generation.
3. Ask with a direct prompt to paste paths, and do not present a suggested file list unless requested.
4. Ask for paths only once per active chat context; reuse the last confirmed paths for follow-up questions unless the user asks to change them.
5. Load documents before splitting them when the corpus changes.
6. Split loaded documents into overlapping chunks before embedding.
7. Create embeddings with the local transformer model before persisting vectors.
8. Query retrieval before drafting any answer that should be grounded in local data.
9. Build the answer prompt from retrieved chunks and keep the response anchored to source text.
10. Every grounded answer must include an evidence section with top-k chunk citations (source path and chunk index).

Implementation notes:

- The MCP server runs over stdio from `python server.py`.
- The RAG backend uses FAISS for vector storage and retrieval, and SentenceTransformers for embeddings.
- Server instructions should stay focused on the exact RAG step each tool performs.

When the user asks for a grounded answer, collect selected file paths once, then use `embed_query`, `query_retrieval`, `build_prompt`, and `generate_answer` for each question, always returning grounded evidence.