# Copilot RAG MCP

A Python-based MCP server that adds retrieval-augmented generation to Copilot in VS Code.
It loads local documents, splits them into chunks, builds embeddings, stores them in FAISS, and returns grounded answers with evidence.

## 🛠️ Tech Stack

- Python 3.12+
- MCP (`mcp`)
- FAISS (`faiss-cpu`)
- SentenceTransformers (`sentence-transformers`)
- NumPy
- PyPDF (`pypdf`)
- Beautiful Soup (`beautifulsoup4`)
- VS Code MCP integration

## ⚙️ What It Does

The server exposes these tools:

- `load_documents`
- `split_documents`
- `create_embeddings`
- `store_vectors`
- `embed_query`
- `query_retrieval`
- `build_prompt`
- `generate_answer`
- `rag_status`

It supports these file types:

- `.txt`
- `.md`
- `.markdown`
- `.html`
- `.htm`
- `.pdf`

## 🧩 VS Code Setup

This workspace already includes MCP registration in [.vscode/mcp.json](.vscode/mcp.json).
VS Code launches the server using the project virtual environment:

```json
{
  "servers": {
    "copilot-rag-mcp": {
      "type": "stdio",
      "command": "D:/Projects IV/Copilot RAG/.venv/Scripts/python.exe",
      "args": ["server.py"]
    }
  }
}
```

### 🚀 Setup Steps

1. Open the workspace in VS Code.
2. Create the virtual environment if it does not exist:

```bash
python -m venv .venv
```

3. Install dependencies:

```bash
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

4. Make sure VS Code is using [.vscode/mcp.json](.vscode/mcp.json).
5. Reload VS Code if the MCP server does not appear immediately.

### 📦 Download the Embedding Model Locally

The server uses `sentence-transformers/all-MiniLM-L6-v2`.
It will download automatically the first time you run the server, then cache it locally.

If you want to pre-download it yourself, run:

```bash
.venv\Scripts\python.exe -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

After that, the model is stored in your local Hugging Face cache and can be reused offline on the same machine.

## ▶️ How to Use It

In Copilot Chat, choose the `copilot-rag-mcp` server and use the tools in order.

Typical workflow:

1. Paste the file path(s) you want to ground the answer on.
2. Run `load_documents` for those paths.
3. Run `split_documents` with the chunk settings you want.
4. Run `create_embeddings`.
5. Run `store_vectors`.
6. Ask your question.
7. Run `query_retrieval` or `build_prompt` with `top_k` and the selected `source_paths`.
8. Run `generate_answer` for the final grounded response.

## 🔧 Defaults

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Default chunk size: `1200`
- Default chunk overlap: `200`
- Default retrieval `top_k`: `5`

## 📝 Notes

- The embedding model is downloaded automatically the first time it is used, then cached locally.
- Retrieved answers should include evidence with source path and chunk index.
- If you want answers from a specific file, pass its path through `source_paths`.
- If the model is not already cached, pre-downloading it with the command above avoids waiting during the first query.

## 📁 Project Files

- [server.py](server.py)
- [requirements.txt](requirements.txt)
- [pyproject.toml](pyproject.toml)
- [.vscode/mcp.json](.vscode/mcp.json)
