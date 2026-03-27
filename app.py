"""
╔══════════════════════════════════════════════════════════════════╗
║          MEDI-BRIDGE — Vernacular GenAI Patient Advocate         ║
║          Team: DNS  |  Lead: Devansh Verma                       ║
║          Stack: Streamlit · Ollama/LLaMA3 · FAISS · OCR          ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pdfplumber
import pytesseract
import pandas as pd
import numpy as np
import faiss
import json
import math
import re
import io
import os
import urllib.request
import urllib.error
from PIL import Image
from datetime import datetime

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medi-Bridge",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# CUSTOM CSS  (clean medical-teal theme)
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0a3d52 0%, #0d5c73 60%, #0e7490 100%);
    color: white;
}
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stTextInput label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stMultiSelect label { color: #a8d8e8 !important; font-size: 0.8rem; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: white !important; }

/* ── hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0a3d52 0%, #0e7490 50%, #06b6d4 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: "🏥";
    position: absolute; right: 2rem; top: 50%;
    transform: translateY(-50%);
    font-size: 5rem; opacity: 0.15;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem; color: white;
    margin: 0; letter-spacing: -0.5px;
}
.hero-sub { color: #a8d8e8; font-size: 1rem; margin-top: 0.4rem; }

/* ── cards ── */
.result-card {
    background: white;
    border: 1px solid #e0f2f7;
    border-left: 4px solid #0e7490;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin: 0.8rem 0;
    box-shadow: 0 2px 12px rgba(14,116,144,0.07);
}
.result-card h3 { color: #0a3d52; font-size: 1rem; margin-bottom: 0.5rem; }
.result-card p  { color: #374151; font-size: 0.95rem; line-height: 1.65; margin: 0; }

/* ── metric boxes ── */
.metric-row { display: flex; gap: 1rem; margin: 1rem 0; flex-wrap: wrap; }
.metric-box {
    flex: 1; min-width: 120px;
    background: linear-gradient(135deg, #f0fdff, #e0f7fa);
    border: 1px solid #b2ebf2;
    border-radius: 10px;
    padding: 0.9rem 1rem;
    text-align: center;
}
.metric-box .val {
    font-size: 1.8rem; font-weight: 700;
    color: #0e7490; line-height: 1;
}
.metric-box .lbl {
    font-size: 0.72rem; color: #6b7280;
    text-transform: uppercase; letter-spacing: 0.06em;
    margin-top: 0.2rem;
}

/* ── warning / disclaimer ── */
.disclaimer-box {
    background: #fffbeb; border: 1px solid #fcd34d;
    border-left: 4px solid #f59e0b;
    border-radius: 10px; padding: 1rem 1.2rem;
    font-size: 0.85rem; color: #78350f;
}
.alert-box {
    background: #fef2f2; border: 1px solid #fca5a5;
    border-left: 4px solid #ef4444;
    border-radius: 10px; padding: 1rem 1.2rem;
    font-size: 0.9rem; color: #7f1d1d; margin: 0.6rem 0;
}
.safe-box {
    background: #f0fdf4; border: 1px solid #86efac;
    border-left: 4px solid #22c55e;
    border-radius: 10px; padding: 1rem 1.2rem;
    font-size: 0.9rem; color: #14532d; margin: 0.6rem 0;
}
/* confidence bar */
.conf-bar-wrap { background: #e5e7eb; border-radius: 999px; height: 10px; margin: 0.4rem 0; }
.conf-bar { height: 10px; border-radius: 999px; background: linear-gradient(90deg, #0e7490, #06b6d4); }

/* ── tab override ── */
.stTabs [data-baseweb="tab"] { font-size: 0.9rem; font-weight: 500; }
.stTabs [aria-selected="true"] { color: #0e7490 !important; border-bottom-color: #0e7490 !important; }

/* ── upload box ── */
[data-testid="stFileUploader"] {
    border: 2px dashed #0e7490 !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}
/* ── badge ── */
.badge {
    display: inline-block;
    background: #e0f7fa; color: #0a3d52;
    border-radius: 999px; padding: 0.15rem 0.7rem;
    font-size: 0.75rem; font-weight: 600; margin: 0.1rem;
}
.badge-red   { background:#fee2e2; color:#7f1d1d; }
.badge-green { background:#dcfce7; color:#14532d; }
.badge-amber { background:#fef9c3; color:#713f12; }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════
# MEDICAL KNOWLEDGE BASE  (RAG corpus — fully local, no license)
# ═════════════════════════════════════════════════════════════════
MEDICAL_KB = [
    # ── Lab values ──
    {"id":"kb001","text":"Hemoglobin (Hb) normal range: Men 13.5–17.5 g/dL, Women 12.0–15.5 g/dL. Low Hb indicates anemia.","tags":["hb","hemoglobin","anemia","blood"]},
    {"id":"kb002","text":"Fasting blood glucose normal: 70–99 mg/dL. Prediabetes: 100–125 mg/dL. Diabetes: ≥126 mg/dL.","tags":["glucose","diabetes","blood sugar","fasting"]},
    {"id":"kb003","text":"HbA1c (Glycated Hemoglobin): Normal <5.7%, Prediabetes 5.7–6.4%, Diabetes ≥6.5%. Reflects average blood sugar over 3 months.","tags":["hba1c","diabetes","glycated"]},
    {"id":"kb004","text":"Creatinine normal: Men 0.74–1.35 mg/dL, Women 0.59–1.04 mg/dL. High creatinine suggests kidney dysfunction.","tags":["creatinine","kidney","renal"]},
    {"id":"kb005","text":"eGFR (estimated Glomerular Filtration Rate): ≥90 normal, 60–89 mildly reduced, 30–59 moderately reduced, <30 severely reduced kidney function.","tags":["egfr","kidney","renal","gfr"]},
    {"id":"kb006","text":"TSH (Thyroid Stimulating Hormone) normal: 0.4–4.0 mIU/L. High TSH = hypothyroidism. Low TSH = hyperthyroidism.","tags":["tsh","thyroid","hormone"]},
    {"id":"kb007","text":"Total Cholesterol: Desirable <200 mg/dL, Borderline 200–239, High ≥240 mg/dL. High cholesterol increases heart disease risk.","tags":["cholesterol","heart","lipids","cardiovascular"]},
    {"id":"kb008","text":"LDL (Bad) Cholesterol: Optimal <100 mg/dL. HDL (Good) Cholesterol: Men >40 mg/dL, Women >50 mg/dL.","tags":["ldl","hdl","cholesterol","lipids"]},
    {"id":"kb009","text":"Triglycerides normal: <150 mg/dL. Borderline 150–199. High 200–499. Very High ≥500 mg/dL.","tags":["triglycerides","lipids","heart"]},
    {"id":"kb010","text":"WBC (White Blood Cell) normal: 4,500–11,000 cells/mcL. High WBC may indicate infection or inflammation. Low WBC may suggest immune problems.","tags":["wbc","white blood cell","infection","immune"]},
    {"id":"kb011","text":"Platelets normal: 150,000–400,000/mcL. Low platelets (thrombocytopenia) increases bleeding risk. High platelets may indicate clotting risk.","tags":["platelets","bleeding","clotting","thrombocytopenia"]},
    {"id":"kb012","text":"Sodium normal: 136–145 mEq/L. Low sodium (hyponatremia) causes confusion, seizures. High sodium (hypernatremia) causes thirst, confusion.","tags":["sodium","electrolyte","hyponatremia","hypernatremia"]},
    {"id":"kb013","text":"Potassium normal: 3.5–5.0 mEq/L. Low potassium (hypokalemia) causes muscle weakness, heart arrhythmias. High potassium (hyperkalemia) is dangerous for the heart.","tags":["potassium","electrolyte","hypokalemia","heart"]},
    {"id":"kb014","text":"ALT (Liver enzyme) normal: 7–56 U/L. High ALT indicates liver damage or disease.","tags":["alt","liver","enzyme","hepatic"]},
    {"id":"kb015","text":"AST (Liver enzyme) normal: 10–40 U/L. Elevated in liver disease, heart attack, or muscle injury.","tags":["ast","liver","enzyme"]},
    {"id":"kb016","text":"Bilirubin total normal: 0.1–1.2 mg/dL. High bilirubin causes jaundice (yellowing of skin/eyes) and may indicate liver or bile duct problems.","tags":["bilirubin","liver","jaundice"]},
    {"id":"kb017","text":"Uric acid normal: Men 3.4–7.0 mg/dL, Women 2.4–6.0 mg/dL. High uric acid can cause gout and kidney stones.","tags":["uric acid","gout","kidney stone"]},
    {"id":"kb018","text":"Vitamin D (25-OH) normal: 20–50 ng/mL. Deficiency (<20 ng/mL) causes bone weakness, fatigue, immune problems.","tags":["vitamin d","bone","deficiency","calcium"]},
    {"id":"kb019","text":"Vitamin B12 normal: 200–900 pg/mL. Low B12 causes anemia, nerve damage, fatigue, and cognitive issues.","tags":["vitamin b12","anemia","nerve","fatigue"]},
    {"id":"kb020","text":"Ferritin normal: Men 24–336 ng/mL, Women 11–307 ng/mL. Low ferritin indicates iron deficiency. High ferritin may indicate inflammation or iron overload.","tags":["ferritin","iron","anemia"]},
    # ── Conditions ──
    {"id":"kb021","text":"Type 2 Diabetes: Chronic condition where blood sugar is too high. Managed with diet, exercise, and medications like Metformin. Complications include kidney, eye, and nerve damage.","tags":["diabetes","type 2","metformin","chronic"]},
    {"id":"kb022","text":"Hypertension (High Blood Pressure): BP ≥130/80 mmHg. Increases risk of stroke, heart attack, and kidney disease. Managed with lifestyle changes and medications.","tags":["hypertension","blood pressure","stroke","heart"]},
    {"id":"kb023","text":"Anemia: Low red blood cells or hemoglobin. Symptoms: fatigue, pallor, shortness of breath. Iron deficiency is the most common cause.","tags":["anemia","iron","hemoglobin","fatigue"]},
    {"id":"kb024","text":"Chronic Kidney Disease (CKD): Gradual loss of kidney function. Staged 1–5 based on eGFR. Avoid NSAIDs, control blood pressure and diabetes.","tags":["ckd","kidney","renal","egfr","chronic"]},
    {"id":"kb025","text":"Hypothyroidism: Underactive thyroid. Symptoms: fatigue, weight gain, cold intolerance. Treated with Levothyroxine.","tags":["hypothyroidism","thyroid","tsh","levothyroxine"]},
    {"id":"kb026","text":"Dyslipidemia: Abnormal lipid levels. Increases cardiovascular risk. Managed with statins, diet, and exercise.","tags":["dyslipidemia","cholesterol","statin","lipids","heart"]},
    # ── WHO / CDC guidelines ──
    {"id":"kb027","text":"WHO guideline: Adults should do at least 150 minutes of moderate-intensity aerobic activity per week for cardiovascular health.","tags":["who","exercise","activity","cardiovascular"]},
    {"id":"kb028","text":"CDC recommendation: Diabetic patients should monitor HbA1c every 3 months if uncontrolled, every 6 months if stable.","tags":["cdc","diabetes","hba1c","monitoring"]},
    {"id":"kb029","text":"ICD-10 E11 — Type 2 Diabetes Mellitus. ICD-10 I10 — Essential Hypertension. ICD-10 N18 — Chronic Kidney Disease.","tags":["icd10","coding","diabetes","hypertension","ckd"]},
    {"id":"kb030","text":"Prior authorization often required for MRI, specialist referrals, and brand-name medications. Clinician must document medical necessity with ICD-10 diagnosis codes.","tags":["prior authorization","icd10","mri","insurance","compliance"]},
    {"id":"kb031","text":"Medical coding compliance: CPT 80053 — Comprehensive Metabolic Panel. CPT 85025 — Complete Blood Count. CPT 83036 — HbA1c. Accurate coding prevents claim denials.","tags":["cpt","coding","compliance","cbc","metabolic panel"]},
    {"id":"kb032","text":"HIPAA compliance: Patient medical data must be de-identified before sharing. All explanations must avoid unauthorized disclosure of PHI.","tags":["hipaa","compliance","privacy","phi"]},
]

# ── TF-IDF style RAG (no sentence-transformers needed) ──────────────
def _tokenize(text: str) -> set:
    return set(re.findall(r'\b\w{3,}\b', text.lower()))

def build_rag_index(kb):
    """Build an inverted token index for lightweight retrieval."""
    index = {}
    for doc in kb:
        tokens = _tokenize(doc["text"] + " " + " ".join(doc.get("tags", [])))
        for tok in tokens:
            index.setdefault(tok, []).append(doc["id"])
    id_map = {doc["id"]: doc for doc in kb}
    return index, id_map

def retrieve(query: str, index: dict, id_map: dict, top_k: int = 5) -> list[dict]:
    q_tokens = _tokenize(query)
    scores = {}
    for tok in q_tokens:
        for doc_id in index.get(tok, []):
            scores[doc_id] = scores.get(doc_id, 0) + 1
    ranked = sorted(scores, key=lambda x: -scores[x])[:top_k]
    return [id_map[did] for did in ranked]

RAG_INDEX, RAG_MAP = build_rag_index(MEDICAL_KB)


# ═════════════════════════════════════════════════════════════════
# OCR / DOCUMENT PARSING
# ═════════════════════════════════════════════════════════════════
def extract_text_from_pdf(uploaded_file) -> str:
    text_parts = []
    with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)
    return "\n".join(text_parts)

def extract_text_from_image(uploaded_file) -> str:
    img = Image.open(io.BytesIO(uploaded_file.read()))
    return pytesseract.image_to_string(img)

def extract_text_from_csv(uploaded_file) -> str:
    df = pd.read_csv(uploaded_file)
    return df.to_string(index=False)

def extract_text_from_excel(uploaded_file) -> str:
    df = pd.read_excel(uploaded_file)
    return df.to_string(index=False)

def extract_text(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    uploaded_file.seek(0)
    if name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    elif name.endswith((".png", ".jpg", ".jpeg")):
        return extract_text_from_image(uploaded_file)
    elif name.endswith(".csv"):
        return extract_text_from_csv(uploaded_file)
    elif name.endswith((".xlsx", ".xls")):
        return extract_text_from_excel(uploaded_file)
    else:
        return uploaded_file.read().decode("utf-8", errors="ignore")


# ═════════════════════════════════════════════════════════════════
# SAFETY FILTER
# ═════════════════════════════════════════════════════════════════
SAFE_KEYWORDS_IN_PROMPT = ["explain", "summarize", "what does", "help understand",
                            "describe", "translate", "mean", "report", "lab", "result"]
UNSAFE_PATTERNS = [
    r"\b(prescri|dosage|how much.*take|stop.*medication|replace.*doctor)\b",
    r"\b(diagnose me|tell me if i have|do i have cancer)\b",
]

def safety_check(text: str) -> tuple[bool, str]:
    lower = text.lower()
    for pat in UNSAFE_PATTERNS:
        if re.search(pat, lower):
            return False, "⚠️ Request may ask for diagnosis or prescription — this is outside Medi-Bridge's scope."
    return True, ""


# ═════════════════════════════════════════════════════════════════
# HALLUCINATION RISK SCORER
# ═════════════════════════════════════════════════════════════════
HIGH_RISK_TERMS = ["cancer", "malignant", "tumor", "hiv", "fatal", "terminal",
                   "surgery", "chemotherapy", "dialysis", "transplant"]
NUMERIC_PATTERN = re.compile(r'\b\d+\.?\d*\s*(mg|mmol|g/dl|iu|u/l|%|mcg|ng/ml|pg/ml|meq/l)\b', re.I)

def hallucination_risk_score(report_text: str, rag_docs: list) -> tuple[float, str]:
    """
    Returns (risk_score 0-1, label).
    Higher = more risky. Based on:
    - Presence of high-risk medical terms
    - Numeric lab values found (anchored by RAG = lower risk)
    - RAG coverage
    """
    risk = 0.0
    found_high_risk = [t for t in HIGH_RISK_TERMS if t in report_text.lower()]
    risk += min(len(found_high_risk) * 0.12, 0.4)

    numbers_found = NUMERIC_PATTERN.findall(report_text)
    if numbers_found:
        # Numbers present but RAG covers them → lower risk
        rag_text = " ".join(d["text"] for d in rag_docs).lower()
        covered = sum(1 for n in numbers_found if any(tok in rag_text for tok in str(n).split()))
        coverage_ratio = covered / max(len(numbers_found), 1)
        risk += (1 - coverage_ratio) * 0.3
    else:
        risk += 0.1  # no numbers → slightly less anchored

    if len(rag_docs) < 2:
        risk += 0.2

    risk = round(min(max(risk, 0.0), 1.0), 2)
    if risk < 0.3:
        label = "🟢 Low"
    elif risk < 0.6:
        label = "🟡 Moderate"
    else:
        label = "🔴 High"
    return risk, label


# ═════════════════════════════════════════════════════════════════
# CONFIDENCE SCORER
# ═════════════════════════════════════════════════════════════════
def confidence_score(rag_docs: list, report_text: str) -> tuple[float, str]:
    base = 0.5
    base += min(len(rag_docs) * 0.08, 0.35)
    if NUMERIC_PATTERN.search(report_text):
        base += 0.1
    base = round(min(base, 0.98), 2)
    pct = int(base * 100)
    return base, f"{pct}%"


# ═════════════════════════════════════════════════════════════════
# OLLAMA LLM CALL  (local HTTP API)
# ═════════════════════════════════════════════════════════════════
OLLAMA_URL = "http://localhost:11434/api/generate"

HINDI_SYSTEM = """
आप एक सहानुभूतिपूर्ण चिकित्सा सहायक हैं जो मरीजों को उनकी रिपोर्ट समझाते हैं।
सरल हिंदी में, Grade-6 स्तर पर उत्तर दें।
कोई नुस्खा या निदान मत करें।
"""

ENGLISH_SYSTEM = """
You are a compassionate medical report explainer for patients with low health literacy.
Explain in simple English at a Grade-6 reading level.
Never diagnose, prescribe, or replace professional medical advice.
Always recommend consulting a doctor.
"""

def call_ollama(prompt: str, language: str = "English", model: str = "llama3") -> tuple[str, bool]:
    system = HINDI_SYSTEM if language == "Hindi" else ENGLISH_SYSTEM
    payload = json.dumps({
        "model": model,
        "prompt": f"[SYSTEM]\n{system}\n\n[USER]\n{prompt}",
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 1024}
    }).encode()
    try:
        req = urllib.request.Request(
            OLLAMA_URL, data=payload,
            headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
            return data.get("response", ""), True
    except Exception as e:
        return f"[Ollama not reachable: {e}]", False


def is_ollama_running() -> bool:
    try:
        urllib.request.urlopen("http://localhost:11434", timeout=3)
        return True
    except:
        return False


# ═════════════════════════════════════════════════════════════════
# DEMO / FALLBACK EXPLAINER (when Ollama is offline)
# ═════════════════════════════════════════════════════════════════
HINDI_DEMO = """
**आपकी रिपोर्ट का सारांश (Demo Mode)**

नमस्ते! यह एक डेमो व्याख्या है। आपकी रिपोर्ट में कुछ महत्वपूर्ण मान मिले हैं जिन पर ध्यान देना जरूरी है।

**मुख्य निष्कर्ष:**
- रक्त शर्करा (Blood Sugar): आपका उपवास रक्त शर्करा स्तर सामान्य सीमा से थोड़ा अधिक हो सकता है।
- हीमोग्लोबिन: यदि यह कम है तो आपको थकान महसूस हो सकती है।
- क्रिएटिनिन: किडनी की कार्यक्षमता का संकेत — सामान्य रहना जरूरी है।

**अगले कदम:**
1. अपने डॉक्टर से इन परिणामों पर चर्चा करें।
2. नियमित व्यायाम और संतुलित आहार लें।
3. खूब पानी पिएं।

⚠️ *यह व्याख्या केवल सामान्य जानकारी के लिए है। कृपया डॉक्टर से परामर्श लें।*
"""

ENGLISH_DEMO = """
**Your Report Summary (Demo Mode)**

Hello! This is a demo explanation since Ollama is not running locally.

**What your report shows:**
Your medical report has been parsed and key lab values have been identified. Here is a plain-language explanation:

- **Blood Sugar / Glucose:** If your fasting glucose is above 126 mg/dL, it may indicate diabetes. Between 100–125 mg/dL suggests prediabetes.
- **Hemoglobin (Hb):** Low hemoglobin causes tiredness and breathlessness — this is called anemia.
- **Creatinine / Kidney:** Elevated creatinine may indicate your kidneys are under stress — drink enough water and avoid painkillers like ibuprofen without a doctor's advice.
- **Cholesterol:** High cholesterol increases heart risk — reduce fried foods and exercise regularly.

**Recommended Next Steps:**
1. Share these results with your doctor at your earliest convenience.
2. Do not stop or change any medication based on this report alone.
3. Stay hydrated and follow a balanced diet.
4. If you feel unwell, visit the nearest clinic.

⚠️ *This explanation is for general awareness only and does not replace professional medical advice.*
"""

def demo_explain(language: str) -> str:
    return HINDI_DEMO if language == "Hindi" else ENGLISH_DEMO


# ═════════════════════════════════════════════════════════════════
# BUILD FULL PROMPT FOR LLM
# ═════════════════════════════════════════════════════════════════
def build_prompt(report_text: str, patient: dict, rag_docs: list, language: str) -> str:
    rag_block = "\n".join([f"- {d['text']}" for d in rag_docs])
    patient_block = (
        f"Patient: {patient.get('age', 'Unknown')} year old {patient.get('gender', 'person')}. "
        f"Known conditions: {', '.join(patient.get('conditions', [])) or 'None mentioned'}."
    )
    if language == "Hindi":
        return f"""
नीचे दी गई चिकित्सा रिपोर्ट को सरल हिंदी में समझाएं।

**रोगी जानकारी:** {patient_block}

**चिकित्सा ज्ञान संदर्भ (RAG):**
{rag_block}

**रिपोर्ट:**
{report_text[:3000]}

कृपया निम्नलिखित अनुभाग में उत्तर दें:
1. मुख्य निष्कर्ष (Main Findings)
2. क्या सामान्य है और क्या नहीं (What is normal vs abnormal)
3. स्वास्थ्य पर प्रभाव (Health implications)
4. अनुशंसित अगले कदम (Recommended next steps)
5. चेतावनी संकेत (Warning signs to watch for)
"""
    else:
        return f"""
Explain the following medical report to the patient in plain English (Grade 6 level).

**Patient context:** {patient_block}

**Medical Knowledge (RAG context):**
{rag_block}

**Report text:**
{report_text[:3000]}

Structure your explanation as:
1. Main Findings (what the report says)
2. What is Normal vs Abnormal
3. What This Means for the Patient's Health
4. Recommended Next Steps
5. Warning Signs to Watch For

Be empathetic, clear, and avoid jargon.
"""


# ═════════════════════════════════════════════════════════════════
# SAMPLE REPORTS
# ═════════════════════════════════════════════════════════════════
SAMPLE_REPORT = """PATHOLOGY REPORT — XYZ Diagnostics
Patient: Anonymous  Date: 2024-01-15

COMPLETE BLOOD COUNT (CBC):
Hemoglobin:        10.2 g/dL       [Normal: 13.5–17.5]  LOW
WBC:               11,500 cells/mcL [Normal: 4500–11000] HIGH
Platelets:         210,000 /mcL    [Normal: 150–400K]   NORMAL

METABOLIC PANEL:
Fasting Glucose:   142 mg/dL       [Normal: 70–99]      HIGH
HbA1c:             7.8%            [Normal: <5.7%]      HIGH
Creatinine:        1.6 mg/dL       [Normal: 0.74–1.35]  HIGH
eGFR:              52 mL/min       [Normal: ≥60]        REDUCED

LIPID PANEL:
Total Cholesterol: 245 mg/dL       [Normal: <200]       HIGH
LDL Cholesterol:   158 mg/dL       [Normal: <100]       HIGH
HDL Cholesterol:   38 mg/dL        [Normal: >40]        LOW
Triglycerides:     210 mg/dL       [Normal: <150]       HIGH

LIVER FUNCTION:
ALT:               62 U/L          [Normal: 7–56]       HIGH
AST:               48 U/L          [Normal: 10–40]      HIGH
Bilirubin:         1.0 mg/dL       [Normal: 0.1–1.2]    NORMAL

VITAMINS & MINERALS:
Vitamin D:         14 ng/mL        [Normal: 20–50]      LOW
Vitamin B12:       180 pg/mL       [Normal: 200–900]    LOW
Ferritin:          8 ng/mL         [Normal: 24–336]     LOW

Reviewed by: Dr. Sample (MD)
"""


# ═════════════════════════════════════════════════════════════════
# STREAMLIT UI — MAIN
# ═════════════════════════════════════════════════════════════════
def main():
    # ── SIDEBAR: Patient Profile ──────────────────────────────────
    with st.sidebar:
        st.markdown("## 🏥 Medi-Bridge")
        st.markdown("*Vernacular GenAI Patient Advocate*")
        st.markdown("---")
        st.markdown("### 👤 Patient Profile")

        age = st.number_input("Age", min_value=1, max_value=120, value=45, step=1)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        conditions_options = [
            "Type 2 Diabetes", "Hypertension", "Chronic Kidney Disease",
            "Heart Disease", "Hypothyroidism", "Asthma", "COPD", "Anemia",
            "Liver Disease", "None"
        ]
        conditions = st.multiselect(
            "Known Conditions",
            options=conditions_options,
            default=["Type 2 Diabetes"]
        )

        st.markdown("---")
        st.markdown("### 🌐 Language")
        language = st.selectbox("Explanation Language", ["English", "Hindi"])

        st.markdown("---")
        st.markdown("### ⚙️ LLM Settings")
        model = st.selectbox("Ollama Model", ["llama3", "llama3.1", "mistral", "phi3"])
        st.caption(f"Endpoint: `localhost:11434`")

        ollama_ok = is_ollama_running()
        if ollama_ok:
            st.success("✅ Ollama: Connected")
        else:
            st.warning("⚠️ Ollama: Not running\nUsing Demo Mode")

        st.markdown("---")
        st.markdown("### 🛡️ Compliance")
        st.markdown(
            '<span class="badge">ICD-10</span>'
            '<span class="badge">CPT</span>'
            '<span class="badge badge-green">HIPAA</span>'
            '<span class="badge badge-green">No PHI stored</span>',
            unsafe_allow_html=True
        )
        st.caption("All processing is local. No data leaves your machine.")

    # ── HERO BANNER ────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">Medi-Bridge</div>
        <div class="hero-sub">
            Vernacular GenAI Patient Advocate · Powered by LLaMA3 (Ollama) · 
            RAG-Grounded · Privacy-Preserving · ICD-10 / CPT Compliant
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── TABS ───────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "📄 Upload & Explain",
        "🔬 Lab Value Analyzer",
        "📚 Medical Knowledge Base"
    ])

    # ══════════════════════════════════════════════════════════════
    # TAB 1 — Upload & Explain
    # ══════════════════════════════════════════════════════════════
    with tab1:
        col_upload, col_sample = st.columns([3, 1])
        with col_upload:
            st.markdown("#### 📤 Upload Medical Report")
            uploaded = st.file_uploader(
                "Supported: PDF, PNG, JPG, CSV, XLSX, TXT",
                type=["pdf", "png", "jpg", "jpeg", "csv", "xlsx", "xls", "txt"]
            )
        with col_sample:
            st.markdown("#### 🧪 Or Use Sample")
            if st.button("Load Sample Report", use_container_width=True):
                st.session_state["report_text"] = SAMPLE_REPORT
                st.success("Sample report loaded!")

        # Extract text from upload
        if uploaded:
            with st.spinner("📖 Extracting text..."):
                try:
                    extracted = extract_text(uploaded)
                    st.session_state["report_text"] = extracted
                    st.success(f"✅ Extracted {len(extracted)} characters from `{uploaded.name}`")
                except Exception as e:
                    st.error(f"Extraction failed: {e}")

        report_text = st.session_state.get("report_text", "")

        if report_text:
            with st.expander("📋 View Extracted Text", expanded=False):
                st.text_area("Raw extracted text", report_text, height=200, disabled=True)

            # ── Safety check ──
            safe, safety_msg = safety_check(report_text)

            # ── RAG retrieval ──
            rag_docs = retrieve(report_text, RAG_INDEX, RAG_MAP, top_k=6)

            # ── Scores ──
            h_risk, h_label = hallucination_risk_score(report_text, rag_docs)
            conf, conf_pct  = confidence_score(rag_docs, report_text)

            # ── Metric row ──
            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-box">
                    <div class="val">{len(rag_docs)}</div>
                    <div class="lbl">RAG Docs Retrieved</div>
                </div>
                <div class="metric-box">
                    <div class="val">{conf_pct}</div>
                    <div class="lbl">Confidence Score</div>
                </div>
                <div class="metric-box">
                    <div class="val">{h_label}</div>
                    <div class="lbl">Hallucination Risk</div>
                </div>
                <div class="metric-box">
                    <div class="val">{'✅' if safe else '⚠️'}</div>
                    <div class="lbl">Safety Filter</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if not safe:
                st.markdown(f'<div class="alert-box">{safety_msg}</div>', unsafe_allow_html=True)

            # ── EXPLAIN button ──
            if st.button("🧠 Generate Patient Explanation", type="primary", use_container_width=True):
                patient = {"age": age, "gender": gender, "conditions": conditions}

                with st.spinner("🔍 Retrieving medical context..."):
                    rag_docs = retrieve(report_text, RAG_INDEX, RAG_MAP, top_k=6)

                with st.spinner("🤖 LLaMA3 generating explanation..."):
                    if ollama_ok:
                        prompt = build_prompt(report_text, patient, rag_docs, language)
                        explanation, success = call_ollama(prompt, language, model)
                    else:
                        explanation = demo_explain(language)
                        success = False

                st.markdown("---")
                st.markdown("### 📝 Patient Explanation")

                lang_badge = "🇮🇳 Hindi" if language == "Hindi" else "🇬🇧 English"
                mode_badge = "🟢 Ollama Live" if (ollama_ok and success) else "🟡 Demo Mode"
                st.markdown(
                    f'<span class="badge">{lang_badge}</span>'
                    f'<span class="badge">{mode_badge}</span>'
                    f'<span class="badge">Age: {age} · {gender}</span>',
                    unsafe_allow_html=True
                )
                st.markdown("")
                st.markdown(f'<div class="result-card"><p>{explanation.replace(chr(10), "<br>")}</p></div>',
                            unsafe_allow_html=True)

                # ── Sources ──
                with st.expander("📚 RAG Source Attribution", expanded=False):
                    for i, doc in enumerate(rag_docs, 1):
                        tags_html = "".join(f'<span class="badge">{t}</span>' for t in doc.get("tags", [])[:4])
                        st.markdown(
                            f'<div class="result-card">'
                            f'<h3>Source {i} · ID: {doc["id"]}</h3>'
                            f'<p>{doc["text"]}</p>'
                            f'{tags_html}'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                # ── Confidence bar ──
                st.markdown("#### 📊 Reliability Metrics")
                st.markdown(
                    f"**Confidence Score:** {conf_pct}"
                    f'<div class="conf-bar-wrap"><div class="conf-bar" style="width:{int(conf*100)}%"></div></div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"**Hallucination Risk:** {h_label} &nbsp; `{int(h_risk*100)}%`"
                )

                # ── Disclaimer ──
                st.markdown(
                    '<div class="disclaimer-box">⚠️ <strong>Medical Disclaimer:</strong> '
                    'Medi-Bridge provides general health information only. It does not diagnose, '
                    'prescribe, or replace professional medical advice. Always consult a qualified '
                    'healthcare provider for medical decisions.</div>',
                    unsafe_allow_html=True
                )

    # ══════════════════════════════════════════════════════════════
    # TAB 2 — Lab Value Analyzer
    # ══════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("#### 🔬 Manual Lab Value Checker")
        st.caption("Enter individual lab values to check if they are within normal range.")

        LAB_REFERENCE = {
            "Hemoglobin (g/dL)":          {"male": (13.5, 17.5), "female": (12.0, 15.5), "unit": "g/dL"},
            "Fasting Glucose (mg/dL)":     {"all":  (70,   99),   "unit": "mg/dL"},
            "HbA1c (%)":                   {"all":  (0,    5.6),  "unit": "%"},
            "Creatinine (mg/dL)":          {"male": (0.74, 1.35), "female": (0.59, 1.04), "unit": "mg/dL"},
            "eGFR (mL/min)":               {"all":  (60,   999),  "unit": "mL/min"},
            "Total Cholesterol (mg/dL)":   {"all":  (0,    199),  "unit": "mg/dL"},
            "LDL Cholesterol (mg/dL)":     {"all":  (0,    99),   "unit": "mg/dL"},
            "HDL Cholesterol (mg/dL)":     {"male": (40,   999),  "female": (50,  999), "unit": "mg/dL"},
            "Triglycerides (mg/dL)":       {"all":  (0,    149),  "unit": "mg/dL"},
            "WBC (cells/mcL)":             {"all":  (4500, 11000),"unit": "cells/mcL"},
            "Platelets (K/mcL)":           {"all":  (150,  400),  "unit": "K/mcL"},
            "ALT (U/L)":                   {"all":  (7,    56),   "unit": "U/L"},
            "AST (U/L)":                   {"all":  (10,   40),   "unit": "U/L"},
            "Vitamin D (ng/mL)":           {"all":  (20,   50),   "unit": "ng/mL"},
            "Vitamin B12 (pg/mL)":         {"all":  (200,  900),  "unit": "pg/mL"},
            "TSH (mIU/L)":                 {"all":  (0.4,  4.0),  "unit": "mIU/L"},
        }

        selected_labs = st.multiselect(
            "Select lab tests to check:",
            options=list(LAB_REFERENCE.keys()),
            default=["Fasting Glucose (mg/dL)", "Hemoglobin (g/dL)", "HbA1c (%)"]
        )

        results_data = []
        for lab in selected_labs:
            ref = LAB_REFERENCE[lab]
            col1, col2 = st.columns([2, 1])
            with col1:
                val = st.number_input(f"{lab}", min_value=0.0, max_value=100000.0,
                                      value=0.0, step=0.1, key=f"lab_{lab}")
            with col2:
                if val > 0:
                    if "male" in ref and "female" in ref:
                        lo, hi = ref["male"] if gender == "Male" else ref["female"]
                    else:
                        lo, hi = ref.get("all", (0, 9999))
                    if lo <= val <= hi:
                        st.markdown('<div class="safe-box">✅ Normal</div>', unsafe_allow_html=True)
                        status = "Normal"
                    elif val < lo:
                        st.markdown(f'<div class="alert-box">⬇️ LOW (Ref: {lo}–{hi})</div>', unsafe_allow_html=True)
                        status = "Low"
                    else:
                        st.markdown(f'<div class="alert-box">⬆️ HIGH (Ref: {lo}–{hi})</div>', unsafe_allow_html=True)
                        status = "High"
                    results_data.append({"Test": lab, "Value": val, "Status": status,
                                         "Normal Range": f"{lo}–{hi} {ref['unit']}"})

        if results_data:
            df = pd.DataFrame(results_data)
            st.markdown("#### 📊 Results Summary")
            st.dataframe(df, use_container_width=True, hide_index=True)

            abnormal = df[df["Status"] != "Normal"]
            if len(abnormal):
                st.markdown(
                    f'<div class="alert-box">⚠️ <strong>{len(abnormal)} abnormal value(s) found.</strong> '
                    f'Please consult your doctor about: {", ".join(abnormal["Test"].tolist())}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="safe-box">✅ All checked values are within normal range.</div>',
                    unsafe_allow_html=True
                )

    # ══════════════════════════════════════════════════════════════
    # TAB 3 — Knowledge Base Browser
    # ══════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("#### 📚 Medical Knowledge Base (RAG Corpus)")
        st.caption(f"Total entries: {len(MEDICAL_KB)} · Fully local, no proprietary data")

        search_q = st.text_input("🔍 Search knowledge base", placeholder="e.g. diabetes, kidney, cholesterol")
        if search_q:
            hits = retrieve(search_q, RAG_INDEX, RAG_MAP, top_k=8)
            st.markdown(f"**{len(hits)} result(s) found for:** `{search_q}`")
            for doc in hits:
                tags_html = "".join(f'<span class="badge">{t}</span>' for t in doc.get("tags", []))
                st.markdown(
                    f'<div class="result-card">'
                    f'<h3>🔖 {doc["id"]}</h3>'
                    f'<p>{doc["text"]}</p>'
                    f'{tags_html}'
                    f'</div>',
                    unsafe_allow_html=True
                )
        else:
            for doc in MEDICAL_KB:
                tags_html = "".join(f'<span class="badge">{t}</span>' for t in doc.get("tags", [])[:5])
                st.markdown(
                    f'<div class="result-card">'
                    f'<h3>🔖 {doc["id"]}</h3>'
                    f'<p>{doc["text"]}</p>'
                    f'{tags_html}'
                    f'</div>',
                    unsafe_allow_html=True
                )

    # ── Footer ──
    st.markdown("---")
    st.markdown(
        '<p style="text-align:center;color:#6b7280;font-size:0.8rem;">'
        '🏥 <strong>Medi-Bridge</strong> · DNS Team · Devansh Verma, Shourya, Nilay Joji · '
        'ETHackathon 2026 · All inference is local · No patient data is stored or transmitted'
        '</p>',
        unsafe_allow_html=True
    )


# ── session state init ──
if "report_text" not in st.session_state:
    st.session_state["report_text"] = ""

if __name__ == "__main__":
    main()
