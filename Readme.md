# RAG from Scratch with Ollama

A hands-on exploration of **Retrieval-Augmented Generation (RAG)**, built in Python notebooks using locally hosted language and embedding models. The project moves from a basic retrieve-then-generate pipeline to more advanced approaches for **query translation, indexing, routing, retrieval, reranking, and answer validation**.

The main LLM is **Llama 3.2** served through **Ollama**; most embedding-based examples use **`nomic-embed-text`**. LangChain connects the components, Chroma provides vector search, and LangSmith supports tracing and debugging of the RAG workflows. The notebooks are experiments and learning exercises, rather than a single deployed application.

## What I explored

| Notebook | Experiments |
| --- | --- |
| [`rag_from_scratch_overview.ipynb`](rag_from_scratch_overview.ipynb) | Basic RAG pipeline; web document loading; text splitting; embeddings; Chroma retrieval; context-grounded generation; token counting and cosine similarity; retrieval top-*k* comparisons. |
| [`rag_from_scratch_Query_Translation.ipynb`](rag_from_scratch_Query_Translation.ipynb) | Multi-query retrieval, RAG Fusion using Reciprocal Rank Fusion (RRF), query decomposition, step-back prompting, and Hypothetical Document Embeddings (HyDE). |
| [`ran_from_scratch_indexing.ipynb`](ran_from_scratch_indexing.ipynb) | Summary-based indexing using `MultiVectorRetriever` to retrieve full source documents; a separate ColBERT late-interaction retrieval experiment using RAGatouille. |
| [`rag_from_scratch_routing.ipynb`](rag_from_scratch_routing.ipynb) | Structured LLM routing between programming-language data-source labels; semantic routing between math and physics prompts using embedding similarity; structured query construction for tutorial search. |
| [`rag_from_scratch_retrieval.ipynb`](rag_from_scratch_retrieval.ipynb) | RAG Fusion; cross-encoder reranking; corrective retrieval with document relevance grading and question rewriting; Self-RAG-style groundedness and usefulness checks; long-context experiments. |

> **Scope:** These are the techniques present in my notebooks. Some are independent demonstrations, not integrated into one end-to-end system. In particular, the routing examples select labels or prompts, and the tutorial query-construction example builds a structured query rather than running a complete video database search.

## How the basic RAG pipeline works

```text
Web pages
   │
   ▼
Load documents → Split into chunks → Generate embeddings
                                            │
                                            ▼
                                        Chroma index
                                            │
User question → Embed question → Retrieve relevant chunks
                                            │
                                            ▼
                              Prompt with retrieved context
                                            │
                                            ▼
                                      Llama 3.2
                                            │
                                            ▼
                                         Answer
```

The foundational example loads an article about LLM-powered autonomous agents, chunks the content, embeds the chunks with `nomic-embed-text`, searches a Chroma vector store, and supplies retrieved content to Llama 3.2 for an answer. Other notebooks modify individual parts of that flow.

## Techniques in more detail

### Query translation

- **Multi-query retrieval:** Generate alternate formulations of a question, retrieve documents for each, and combine unique results.
- **RAG Fusion:** Retrieve results for several generated questions and merge rankings using **Reciprocal Rank Fusion (RRF)**, where a document gains score according to its position in each result list.
- **Decomposition:** Break a complex question into sub-questions, answer each with retrieval, and synthesize the partial answers.
- **Step-back prompting:** Generate a broader conceptual question to retrieve complementary background context alongside context for the original question.
- **HyDE:** Generate a hypothetical answer-like passage with the LLM, embed that passage, and retrieve real documents similar to it.

### Indexing

- **Summary indexing / multi-vector retrieval:** Generate summaries of source documents using Llama 3.2, index the summaries in Chroma, and use document IDs to recover the original full documents from an in-memory store.
- **ColBERT experiment:** Use the pretrained `colbert-ir/colbertv2.0` model via RAGatouille to index and search a Wikipedia article, then expose it as a LangChain retriever. This is a **separate pretrained retrieval model**, not an Ollama embedding model.

### Routing and query construction

- **Logical routing:** Ask Llama 3.2 for a structured `RouteQuery` output choosing `python_docs`, `js_docs`, or `golang_docs`, then map that label to a route placeholder.
- **Semantic routing:** Compare the question embedding with math and physics prompt embeddings using cosine similarity, and use the more similar prompt for generation.
- **Query construction:** Define a Pydantic schema for tutorial-search terms and optional metadata filters such as publication date, video length, and view count. The notebook also loads a YouTube transcript as a data-loading example.

### Retrieval quality and answer checks

- **RRF-based retrieval:** Combine results from multiple searches before passing the top-ranked context to the LLM.
- **Cross-encoder reranking:** Rerank retrieved question-document pairs with `cross-encoder/ms-marco-MiniLM-L-6-v2` from Sentence Transformers.
- **Corrective RAG (CRAG-style):** Grade document relevance with structured LLM outputs; rewrite the question and retrieve again when some documents are irrelevant; deduplicate and use the relevant context.
- **Self-RAG-style checks:** Assess whether a generated answer is supported by retrieved passages and whether it answers the question; retry generation a limited number of times if checks fail.
- **Long-context exploration:** Compare answers under short and longer retrieved contexts, and test how moving a target fact to the beginning, middle, or end of a synthetic context affects retrieval from context.

These are notebook-level demonstrations. The grading prompts use an LLM's own judgments, not an independent benchmark for factual correctness.

## Tech stack

**Language and environment:** Python, Jupyter Notebook, VS Code

**Local models (Ollama):**
- `llama3.2` — text generation and structured output
- `nomic-embed-text` — text embeddings

**Core libraries:** LangChain (`langchain-core`, `langchain-community`, `langchain-ollama`, `langchain-chroma`, `langchain-text-splitters`, and `langchain-classic`), Chroma, Beautiful Soup, Pydantic, NumPy

**Additional retrieval tools:** Sentence Transformers (cross-encoder), RAGatouille / ColBERT

**Observability:** LangSmith — optional tracing and debugging of LangChain/LLM workflows

**Other tools used in individual examples:** `tiktoken`, `requests`, and YouTube transcript loading

## LangSmith tracing

I also explored **LangSmith** alongside the Ollama-based RAG notebooks. LangSmith is an observability platform for inspecting LLM application runs; it complements LangChain rather than replacing the local model or vector database.

With tracing enabled, it can help me:

- Follow the steps of a RAG chain, including prompt preparation, model calls, and retrieval operations that are instrumented.
- Inspect prompts, responses, intermediate steps, and errors while debugging notebook experiments.
- Compare how changes to retrieval or prompting affect the execution flow.

The notebooks configure LangSmith tracing, and the Query Translation notebook also uses the LangSmith client to obtain a RAG prompt. **The language model and embeddings still run locally through Ollama**; LangSmith is a separate hosted service, so trace data can be sent to LangSmith when tracing is enabled.

To enable tracing for your own runs, create a LangSmith API key and set the environment variables in your terminal before starting Jupyter or VS Code:

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY="your_langsmith_api_key"
export LANGSMITH_PROJECT="rag-from-scratch-ollama"
```

Tracing is optional for the core Ollama RAG pipeline. **Do not commit API keys or a `.env` file containing secrets to GitHub.** Avoid tracing sensitive document content unless you have reviewed the data-sharing implications.

## Getting started

### 1. Prerequisites

- Python with Jupyter support
- [Ollama](https://ollama.com/) installed and running locally
- Internet access for the web pages and external pretrained models used in the notebooks

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows, activate with `.venv\Scripts\activate` instead.

### 3. Install Python packages

```bash
pip install jupyter ipykernel \
  langchain langchain-core langchain-community langchain-classic \
  langchain-ollama langchain-chroma langchain-text-splitters \
  langsmith chromadb beautifulsoup4 tiktoken numpy pydantic requests \
  sentence-transformers youtube-transcript-api pytube
```

For the optional **ColBERT** indexing notebook, also install:

```bash
pip install ragatouille
```

The notebooks were developed interactively and also contain some in-notebook install commands. Dependency compatibility may vary with Python and library versions; the command above is a package inventory, **not a tested lockfile**.

### 4. Download the local models

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
ollama list
```

Ensure the Ollama service is available when running cells that instantiate `ChatOllama` or `OllamaEmbeddings`.

### 5. Open the notebooks

```bash
jupyter notebook
```

You can also open the `.ipynb` files directly in VS Code and select the Python interpreter from `.venv`. Start with `rag_from_scratch_overview.ipynb` to understand the basic pipeline, then explore the other notebooks by topic. Some cells depend on objects created earlier in the same notebook, so run them in order.

## Notes and current limitations

- The notebooks are **educational experiments**, not a unified API, UI, or production deployment.
- The web-document and Wikipedia examples depend on external content being reachable. ColBERT and the cross-encoder download separate pretrained model weights.
- The routing notebook's tutorial-search section defines a `TutorialSearch` schema, but its later `query_analyzer` uses the previously defined `structured_llm` object, which was configured with `RouteQuery`. That section needs to bind the model to `TutorialSearch` to consistently produce the intended fields.
- The corrective retrieval example assumes the rewritten-query branch has run when it accesses `new_docs` in the next cell. A reusable version would handle the no-rewrite path explicitly.
- LangSmith tracing requires a separate account and API key when enabled; it is not required for local Ollama inference.
- No formal benchmark, latency comparison, accuracy score, or deployed endpoint is provided in these notebooks. The printed examples should not be interpreted as measured improvement over a baseline.

## Project layout

```text
.
├── README.md
├── rag_from_scratch_overview.ipynb
├── rag_from_scratch_Query_Translation.ipynb
├── ran_from_scratch_indexing.ipynb
├── rag_from_scratch_routing.ipynb
└── rag_from_scratch_retrieval.ipynb
```

The filenames above match the notebooks provided, including the current `ran_from_scratch_indexing.ipynb` spelling.

## Background and attribution

This is my **Ollama-based implementation and exploration** of RAG techniques while learning from the broader *RAG from Scratch* tutorial material. The original reference repository is **not my original work**, and this README describes only the code and experiments present in my own five notebooks. It does not claim ownership of the reference materials or of the externally downloaded pretrained models.
