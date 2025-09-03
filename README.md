# PaperPilot Backend

This backend provides document ingestion, vector database creation, and semantic search using free, local embeddings.

## Features

- Loads markdown files from `data/books`
- Splits documents into chunks for embedding
- Uses HuggingFace's `sentence-transformers/all-MiniLM-L6-v2` for free embeddings
- Stores embeddings in a local Chroma vector database
- Provides a CLI for semantic search over your documents

## Setup

1. **Install dependencies:**
   ```bash
   pip install langchain-community langchain-huggingface langchain-chroma sentence-transformers
   ```

2. **Prepare your data:**
   - Place your `.md` files in the `data/books` directory.

3. **Create the vector database:**
   ```bash
   python createdb.py
   ```

4. **Query your data:**
   ```bash
   python query_data.py "Your question here"
   ```

## Files

- `createdb.py` — Loads and embeds documents, creates the Chroma vector store.
- `query_data.py` — CLI tool for semantic search over your documents.
- `data/books/` — Directory for your markdown files.
- `chroma/` — Directory for the vector database (auto-created).

## Notes

- No OpenAI API key required; all embeddings are free and local.
- For best results, ensure your markdown files are well-formatted and relevant to your queries.

## Example

```bash
python query_data.py "How does Alice meet the Mad Hatter?"
```

## License