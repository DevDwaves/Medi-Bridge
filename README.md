# 🏥 Medi-Bridge — Vernacular GenAI Patient Advocate
**Team:** DNS | **Lead:** Devansh Verma | Shourya, Nilay Joji

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```
> Also install Tesseract OCR:
> - Ubuntu/Debian: `sudo apt install tesseract-ocr`
> - macOS: `brew install tesseract`
> - Windows: https://github.com/UB-Mannheim/tesseract/wiki

### 2. Start Ollama with LLaMA3 (for live AI explanations)
```bash
# Install Ollama: https://ollama.com
ollama pull llama3
ollama serve   # starts on localhost:11434
```
> ⚠️ If Ollama is not running, the app runs in **Demo Mode** with pre-written explanations.

### 3. Run the app
```bash
streamlit run app.py
```

---

## 🎯 Features

| Feature | Status |
|---|---|
| PDF/Image/CSV/Excel/TXT upload + OCR | ✅ |
| Patient profile (age, gender, conditions) | ✅ |
| RAG with 32-entry medical knowledge base | ✅ |
| LLaMA3 via Ollama (fully offline) | ✅ |
| Hindi & English vernacular output | ✅ |
| Hallucination risk scoring | ✅ |
| Confidence scoring | ✅ |
| Source attribution (RAG docs) | ✅ |
| Medical safety guardrails | ✅ |
| ICD-10 / CPT / HIPAA compliance notes | ✅ |
| Lab value manual checker | ✅ |
| Knowledge base browser + search | ✅ |
| Demo mode (no Ollama needed) | ✅ |

---

## 🏗️ Architecture

```
User Upload (PDF/IMG/CSV/XLSX)
        ↓
  OCR + Text Extraction
  (pdfplumber + pytesseract)
        ↓
  RAG Retrieval (TF-IDF over local KB)
  [32 medical facts, WHO/CDC guidelines, ICD-10/CPT codes]
        ↓
  Patient-conditioned prompt builder
  [age + gender + known conditions + RAG context]
        ↓
  Ollama → LLaMA3 (offline local inference)
        ↓
  Safety Filter + Hallucination Scorer + Confidence Scorer
        ↓
  Patient-friendly output in Hindi / English
```

---

## 🛡️ Compliance & Safety
- No data stored or transmitted (fully local)
- No diagnosis or prescription generation
- Mandatory medical disclaimers
- HIPAA-aligned: no PHI retention
- ICD-10/CPT code awareness in knowledge base
- Safety filter blocks prescription/diagnosis requests

---

## 📋 Sample Report
Click **"Load Sample Report"** in the app to test with a built-in pathology report 
covering CBC, metabolic panel, lipid panel, liver function, and vitamin levels.
