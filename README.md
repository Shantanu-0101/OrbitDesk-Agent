# OrbitDesk Support Agent.

An AI agent network that answers support questions for OrbitDesk using a local-first architecture with LangGraph and Hugging Face models.

> **AI Disclosure**: This project was built with the assistance of an AI coding assistant (Claude/Antigravity). All code, design decisions, and architecture are understood and can be explained by the author.

---

## 🏗️ Architecture

The agent uses a **LangGraph graph** with 4 main nodes:

```
User Question
     │
     ▼
┌─────────┐
│  TRIAGE │  ──── Classifies: answerable / clarification / escalation / out_of_scope
└────┬────┘
     │
     ▼
┌───────────┐
│ RETRIEVAL │  ──── Embeds query, finds top-K passages from KB + resolved cases
└─────┬─────┘
      │
      ▼
┌──────────────────┐
│ RESPONSE GENERA- │  ──── Local LLM generates answer from retrieved evidence only
│      TION        │
└────────┬─────────┘
         │
         ▼
┌──────────────┐      ┌──────────────┐
│ VERIFICATION │─fail─▶   REVISION   │  ──── Revises once, or returns safe_failure
└──────┬───────┘      └──────────────┘
       │ pass
       ▼
   Final Output (JSON + readable response)
```

---

## 📦 Project Structure

```
orbitdesk-agent/
├── agent/
│   ├── state.py          # Shared typed state (TypedDict)
│   ├── loader.py         # Knowledge base + resolved cases loader
│   ├── retriever.py      # Embedding-based retrieval (FAISS)
│   ├── nodes.py          # All 4 LangGraph nodes
│   ├── graph.py          # LangGraph graph definition
│   └── models.py         # Local model loading (HF transformers)
├── tests/
│   └── test_routing.py   # Automated routing tests
├── outputs/              # Sample run outputs (JSON)
├── diagrams/             # Graph diagram (PNG)
├── main.py               # CLI entry point
├── requirements.txt
└── README.md
```

---

## 🚀 Setup & Installation

### 1. Clone / open the project
```bash
cd orbitdesk-agent
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. First run (downloads models ~500MB, one-time only)
```bash
python main.py
```

> After initial download, you can disconnect the internet and run completely offline.

---

## 🤖 Models Used

| Role | Model | Size |
|------|-------|------|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | ~90MB |
| Generation | `Qwen/Qwen2.5-0.5B-Instruct` | ~500MB |

- **Hardware used**: Intel CPU, 16GB RAM, No GPU (CPU inference)
- **Approx. model load time**: ~15–30 seconds
- **Approx. response latency**: ~10–30 seconds per query (CPU)

---

## ▶️ Running the Agent

```bash
# Run all 5 sample questions
python main.py

# Ask a custom question
python main.py --question "Can a Viewer create an API credential?"

# Run automated tests
python -m pytest tests/test_routing.py -v
```

---

## 🧪 Test Cases Covered

| # | Type | Question |
|---|------|----------|
| 1 | Answerable | Timezone change broke exports |
| 2 | Multi-document | render_failed twice — escalation |
| 3 | Requires clarification | "Data sync not working" |
| 4 | Out of scope | "Issue a refund" |
| 5 | Verification fails → retry | Ambiguous case triggers revision |

---

## ⚙️ Design Trade-offs & Limitations

**Trade-off**: Used `all-MiniLM-L6-v2` for embeddings (fast, small) over larger models. Retrieval quality is good for this KB size but wouldn't scale to thousands of documents without a proper ANN index.

**Known limitation**: The local generation model (Qwen 0.5B) sometimes produces verbose or slightly off-format answers, which is caught by the verification node and triggers revision.

**What I'd improve with more time**: Add a reranker model (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) between retrieval and generation for better passage selection.
