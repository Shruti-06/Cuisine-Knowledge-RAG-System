# South Asian Cuisine RAG System

A Retrieval-Augmented Generation (RAG) system that answers natural-language questions about South Asian cuisine, built end-to-end: corpus construction, chunking, embedding, hybrid retrieval with reranking, grounded LLM generation, benchmark evaluation, and an interactive Gradio UI.

## Highlights

- **411-document corpus** (~330K words) built from Wikipedia, Wikibooks, and food-blog sources, each with a dedicated cleaning pipeline
- **Hybrid retrieval**: dense (FAISS) + BM25 keyword search, fused with weighted scoring (α = 0.6 dense, β = 0.4 BM25)
- **Cross-encoder reranking** for precision at the top of the ranking
- **Grounded generation** with prompt-strategy comparison (zero-shot, few-shot, Chain-of-Thought)
- **250-query benchmark dataset** spanning factual, method, ingredient, regional, and comparison question types
- Best retrieval performance: **nDCG@5 = 0.898** (cross-encoder reranking); best generation: **BERTScore ≈ 0.81**

## Architecture

```
                ┌──────────────────────────────────────────┐
                │              CORPUS (411 docs)           │
                │   Wikipedia (202) · Wikibooks (203)      │
                │   Food blog (6)                          │
                └───────────────────┬──────────────────────┘
                                    │  adaptive source-aware chunking
                                    ▼
                ┌──────────────────────────────────────────┐
                │   Embeddings: BAAI/bge-large-en-v1.5     │
                │   Vector index: FAISS (IndexFlatIP)      │
                └───────────────────┬──────────────────────┘
                                    │
              query ──► hybrid retrieval (dense ⊕ BM25) ──► cross-encoder
                        α = 0.6 / β = 0.4                   reranking
                                    │                       (ms-marco-MiniLM-L-6-v2)
                                    ▼
                ┌──────────────────────────────────────────┐
                │   Grounded generation                    │
                │   Qwen/Qwen2.5-0.5B-Instruct             │
                │   (zero-shot · few-shot · CoT prompts)   │
                └───────────────────┬──────────────────────┘
                                    ▼
                            Gradio interface
```

## Pipeline Components

### 1. Corpus construction
Three source-specific collection and cleaning pipelines:
- **Wikipedia** - MediaWiki API crawl from seed articles, with section splitting, citation-marker removal, and Unicode normalisation
- **Wikibooks** - manual seed pages plus category-based discovery, with recipe-structure-aware cleaning
- **Food blog** - BeautifulSoup scraping with keyword filtering, retry/backoff handling, and boilerplate removal

Each document carries metadata (`title`, `url`, `source`, `section_title`, word counts) used later for source-aware chunking.

### 2. Chunking
Three strategies implemented and compared:
- Fixed-size chunking
- Recursive chunking
- **Adaptive source-aware chunking** (final choice) - chunk boundaries respect the structure of each source type (encyclopedia sections vs. recipes vs. blog posts), with context enrichment (title/section prepended to each chunk)

### 3. Embedding
Three embedding models benchmarked on retrieval quality against the benchmark queries:
- `all-MiniLM-L6-v2`
- **`BAAI/bge-large-en-v1.5`** (final choice)
- `intfloat/multilingual-e5-large`

### 4. Retrieval
- Dense retrieval over a FAISS `IndexFlatIP` index
- BM25 keyword retrieval
- **Weighted hybrid fusion** (α = 0.6 dense, β = 0.4 BM25)
- **Cross-encoder reranking** with `cross-encoder/ms-marco-MiniLM-L-6-v2` (three rerankers compared)

### 5. Generation & evaluation
- Grounded prompting of `Qwen/Qwen2.5-0.5B-Instruct` with retrieved context
- Prompt strategies compared: zero-shot, few-shot, Chain-of-Thought
- Metrics: **nDCG@5** (retrieval), **Exact Match, Token F1, ROUGE-L, BERTScore** (generation)

### 6. Interface
A Gradio UI supporting two input routes: batch JSON upload and free-text questions, returning grounded answers with retrieved context.

## Results

| Component | Best configuration | Metric |
|---|---|---|
| Retrieval | Hybrid + cross-encoder reranking | nDCG@5 = 0.898 |
| Generation | Few-shot prompting | BERTScore ≈ 0.81 |
| Generation | Chain-of-Thought prompting | BERTScore ≈ 0.79 |

*(Results from benchmark evaluation over the 250-query dataset; see the evaluation sections of the notebook.)*

## Repository Structure

```
├── Data/
│   ├── Background_Corpus.json        # Cleaned corpus (411 documents)
│   └── RAG_Benchmark_Dataset.json    # 250-query QA benchmark
├── Notebook/
│   └── Codebase.ipynb                # Full pipeline: corpus → chunking → retrieval → generation → UI
└── README.md
```

## How to Run

### 1. Get the code

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

Or open the notebook directly in **Google Colab** (recommended - a GPU runtime speeds up embedding generation and cross-encoder reranking considerably: `Runtime → Change runtime type → GPU`).

### 2. Install dependencies

```bash
pip install sentence-transformers faiss-cpu rank-bm25 transformers accelerate \
            gradio beautifulsoup4 requests rouge-score bert-score pandas numpy
```

(On Colab, the notebook's **Section 1 - Requirement Setup** installs these for you.)

### 3. Run the notebook

Open the notebook:

```bash
jupyter notebook Notebook/Codebase.ipynb
```

Then run the notebook cells in order. The notebook covers corpus creation, chunking, embeddings, FAISS indexing, retrieval, reranking, generation, evaluation, and the final Gradio interface.

## Example Use Case

A user can ask a question such as:

> **You:** what is tea?

**Output:**

> **Assistant:** Tea is a beverage made from the leaves or buds of plants such as Camellia sinensis, which can be green, black, or oolong. It is commonly consumed in various forms including loose leaf, iced, and hot beverages.

For each question, the system searches the South Asian cuisine corpus for the most relevant chunks, reranks them with a cross-encoder, and passes the top results to the LLM - so every answer is grounded in the retrieved context rather than the model's own memory.

### 4. Use the app

Section 9 launches a **Gradio interface** (a local/shareable URL is printed) with two input routes:

- **Free-text question** - e.g. *"Where is biryani believed to have originated?"* - returns a grounded answer with the retrieved context
- **JSON upload** - batch mode: upload a file of queries and download the answered payload

### Hardware notes

- Runs on CPU, but embedding 411 documents and reranking is slow - expect a GPU (Colab T4 is enough) for a comfortable first run
- `Qwen2.5-0.5B-Instruct` is small enough to run on free-tier Colab

## Acknowledgements

Corpus text is drawn from Wikipedia and Wikibooks (CC BY-SA) and a public food blog. This repository is for academic and portfolio purposes.
