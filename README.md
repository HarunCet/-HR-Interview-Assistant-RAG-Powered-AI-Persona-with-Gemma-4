# 🎯 HR Interview Assistant — RAG-Powered AI Persona With Gemma 4 

> Upload your CV, build your AI persona, and let it answer HR interview questions on your behalf — powered by Gemma 4 and FAISS.

---

## 📸 Screenshots

### Profile Setup Screen


### Interview Chat Screen


---

## 🧠 What It Does

This project lets a user input their personal background — or upload a CV as PDF — and creates an AI persona that answers HR interview questions **in first person**, as if the user themselves is speaking. It uses Retrieval-Augmented Generation (RAG) to ground every answer in the user's actual information, avoiding hallucinations.

**Typical flow:**
1. Enter your name and personal background, or upload your CV (PDF)
2. The system chunks and indexes your information into a FAISS vector database
3. Ask any HR question — the system retrieves the most relevant chunks and feeds them to Gemma 4
4. The model answers as you, in a natural and sincere tone

---

## 🏗️ Architecture

```
User Input (text / PDF)
        │
        ▼
  Text Chunking (80 words, 20-word overlap)
        │
        ▼
  Embedding (multilingual-e5-small)
        │
        ▼
  FAISS Vector Index (IndexFlatIP)
        │
   HR Question
        │
        ▼
  Query Embedding → Top-K Retrieval (k=4)
        │
        ▼
  Prompt Construction (RAG context + persona)
        │
        ▼
  Gemma 4 E2B-IT (4-bit quantized, BitsAndBytes)
        │
        ▼
  Streaming Response (TextIteratorStreamer)
        │
        ▼
  Gradio Chat UI
```

---

## ⚙️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language Model | `google/gemma-4-E2B-it` |
| Quantization | BitsAndBytes (4-bit NF4, bfloat16) |
| Embedding Model | `intfloat/multilingual-e5-small` |
| Vector Database | FAISS (`IndexFlatIP`) |
| PDF Parsing | PyMuPDF (fitz) |
| Streaming | `TextIteratorStreamer` + Python Threading |
| UI | Gradio (2-screen flow) |
| Runtime | Google Colab (T4 GPU) |

---

## 📊 Evaluation Results

Evaluated on 5 standard HR questions using a real CV as input.

| Metric | Score | Status |
|--------|-------|--------|
| Retrieval Similarity (avg) | **0.809** | ✅ Excellent |
| Answer Relevance (avg) | **0.855** | ✅ Excellent |
| Semantic Faithfulness — max (avg) | **0.878** | ✅ Excellent |
| Semantic Faithfulness — mean (avg) | **0.864** | ✅ Excellent |
| Retrieval Latency | **~0.013s** | ✅ Near-instant |
| Generation Speed | **~8.3 tok/s** | ✅ T4 optimal |
| End-to-End Response Time | **~17–21s** | ✅ Acceptable |

### Per-Question Breakdown

| Question | Retrieval | Relevance | Sem. Faithfulness (max) | Speed |
|----------|-----------|-----------|--------------------------|-------|
| Tell me about yourself | 0.817 | 0.848 | 0.915 | 8.4 tok/s |
| Strengths & weaknesses | 0.803 | 0.880 | 0.883 | 8.2 tok/s |
| Why change jobs? | 0.814 | 0.851 | 0.859 | 8.3 tok/s |
| Where in 5 years? | 0.799 | 0.836 | 0.876 | 8.0 tok/s |
| Team management exp? | 0.812 | 0.859 | 0.857 | 8.4 tok/s |

### Evaluation Metrics Explained

- **Retrieval Similarity** — Cosine similarity between the HR question and retrieved chunks (via `multilingual-e5-small`). Measures whether the right context is being fetched.
- **Answer Relevance** — Semantic similarity between the question and the generated answer. Measures whether the model actually addresses the question.
- **Semantic Faithfulness** — Cosine similarity between the generated answer and the retrieved chunks. Measures whether the answer is grounded in the user's actual information rather than hallucinated.

---

## 🚀 Setup & Usage

### 1. Install Dependencies

```python
!pip install -q transformers accelerate bitsandbytes
!pip install -q faiss-cpu sentence-transformers
!pip install -q gradio pymupdf
!pip install -q rouge-score
```

### 2. Load the Model

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "google/gemma-4-E2B-it"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
```

### 3. Run the Notebook

Execute cells in order:
- **Cell 3** — Embedding model + FAISS setup
- **Cell 4** — Prompt builder + answer generator
- **Cell 5** — Enter user info and build vector DB
- **Cell 6** — Single question test
- **Cell 7** — Interactive loop (no UI)
- **Cell 8** — Gradio UI (2-screen flow)
- **Cell 9–12** — Evaluation pipeline

### 4. Launch the UI

```python
demo.launch(share=True, debug=True)
```

Open the public link, upload your CV or paste your background, click **Mülakata Başla** and start asking HR questions.

---

## 🗂️ Project Structure

```
hr-interview-assistant/
│
├── notebook.ipynb          # Main Colab notebook (all cells)
├── README.md               # This file
└── screenshots/
    ├── profile_screen.png
    └── chat_screen.png
```

---

## 🔧 Key Design Decisions

**Why RAG instead of fine-tuning?**
Fine-tuning requires labeled data and compute. RAG lets any user plug in their own CV instantly, with no training required.

**Why FAISS over a hosted vector DB?**
This runs entirely on Colab with no external dependencies or API keys needed.

**Why `multilingual-e5-small`?**
It supports both Turkish and English out of the box, uses `query:/passage:` prefixes for better retrieval precision, and is lightweight enough to run alongside the LLM on a T4.

**Why 4-bit quantization?**
Gemma 4 E2B in full precision exceeds T4 VRAM. NF4 quantization reduces memory ~4x with minimal quality loss.

---

## 📈 Future Improvements

- **Interview Mode** — System automatically asks 10 classic HR questions in sequence and gives a final evaluation
- **Voice Input** — Accept spoken questions via `gr.Audio`
- **PDF Report** — Export the full interview session as a downloadable PDF

---

## 📄 License

MIT License — free to use, modify and distribute.

---

## 🙏 Acknowledgements

- [Google Gemma](https://ai.google.dev/gemma) for the open-weight model
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [FAISS](https://github.com/facebookresearch/faiss) by Meta AI
- [Gradio](https://gradio.app/)
