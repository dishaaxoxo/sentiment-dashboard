"""
AI Sentiment Analysis — FastAPI Backend
========================================
HOW TO RUN:
  1. Open terminal / command prompt
  2. Run: pip install fastapi uvicorn transformers torch sqlalchemy python-multipart
  3. Run: uvicorn app:app --reload
  4. Server starts at: http://localhost:8000
  5. API docs at:      http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import torch
import os
import re
import sqlite3
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_PATH = "./sentiment_model"   # folder saved by notebook
DB_PATH    = "./feedback.db"
LABELS     = {0: "Negative", 1: "Neutral", 2: "Positive"}
device     = "cuda" if torch.cuda.is_available() else "cpu"

# ── FASTAPI APP ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="SentimentAI API",
    description="Real-time customer feedback sentiment analysis using DistilBERT",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── DATABASE SETUP ────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            raw_text   TEXT    NOT NULL,
            clean_text TEXT,
            label      TEXT    NOT NULL,
            confidence REAL    NOT NULL,
            source     TEXT    DEFAULT 'api',
            created_at TEXT    NOT NULL
        )
    """)
    conn.commit(); conn.close()

def save_to_db(raw_text, clean_text, label, confidence, source="api"):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO feedback (raw_text, clean_text, label, confidence, source, created_at) VALUES (?,?,?,?,?,?)",
        (raw_text, clean_text, label, confidence, source, datetime.utcnow().isoformat())
    )
    conn.commit(); conn.close()

# ── MODEL LOADING ─────────────────────────────────────────────────────────────
tokenizer = None
model     = None

def load_model():
    global tokenizer, model
    if os.path.exists(MODEL_PATH):
        print(f"✅ Loading fine-tuned model from {MODEL_PATH}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    else:
        print("⚠️  Fine-tuned model not found — loading base distilbert-sst2")
        print("   Run the Jupyter notebook first to generate ./sentiment_model/")
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        model     = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    model.to(device)
    model.eval()
    print(f"✅ Model loaded on {device.upper()}")

# ── TEXT CLEANING ─────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^\w\s\'\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ── INFERENCE ─────────────────────────────────────────────────────────────────
def run_inference(text: str):
    cleaned = clean_text(text)
    enc = tokenizer(
        cleaned, max_length=128, padding='max_length',
        truncation=True, return_tensors='pt'
    )
    with torch.no_grad():
        out = model(
            input_ids=enc['input_ids'].to(device),
            attention_mask=enc['attention_mask'].to(device)
        )
    probs = torch.softmax(out.logits, dim=1).cpu().numpy()[0]
    pred  = int(probs.argmax())
    # Map 2-class SST-2 to 3-class if using base model
    if len(probs) == 2:
        if probs[1] > 0.75:
            label = "Positive"; conf = float(probs[1])
        elif probs[0] > 0.75:
            label = "Negative"; conf = float(probs[0])
        else:
            label = "Neutral"; conf = float(max(probs[0], probs[1]))
    else:
        label = LABELS[pred]
        conf  = float(probs[pred])
    return label, round(conf * 100, 2), cleaned

# ── REQUEST / RESPONSE MODELS ─────────────────────────────────────────────────
class TextRequest(BaseModel):
    text: str
    source: Optional[str] = "api"

class BatchRequest(BaseModel):
    texts: List[str]
    source: Optional[str] = "batch"

class AnalysisResponse(BaseModel):
    text: str
    label: str
    confidence: float
    timestamp: str

# ── ENDPOINTS ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup():
    init_db()
    load_model()

@app.get("/")
def root():
    """Serve dashboard if available, else show API info."""
    if os.path.exists("dashboard.html"):
        return FileResponse("dashboard.html")
    return {
        "message": "SentimentAI API is running",
        "endpoints": ["/analyze", "/batch", "/trends", "/history", "/docs"]
    }

@app.post("/analyze", response_model=AnalysisResponse)
def analyze(req: TextRequest):
    """Analyze sentiment of a single text."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    label, conf, cleaned = run_inference(req.text)
    ts = datetime.utcnow().isoformat()
    save_to_db(req.text, cleaned, label, conf, req.source)
    return AnalysisResponse(text=req.text, label=label, confidence=conf, timestamp=ts)

@app.post("/batch")
def batch_analyze(req: BatchRequest):
    """Analyze multiple texts at once."""
    if not req.texts:
        raise HTTPException(status_code=400, detail="No texts provided")
    results = []
    for text in req.texts[:50]:   # cap at 50
        label, conf, cleaned = run_inference(text)
        ts = datetime.utcnow().isoformat()
        save_to_db(text, cleaned, label, conf, req.source)
        results.append({"text": text, "label": label, "confidence": conf, "timestamp": ts})
    summary = {
        "Positive": sum(1 for r in results if r["label"]=="Positive"),
        "Negative": sum(1 for r in results if r["label"]=="Negative"),
        "Neutral":  sum(1 for r in results if r["label"]=="Neutral"),
    }
    return {"total": len(results), "summary": summary, "results": results}

@app.get("/trends")
def get_trends(days: int = 7):
    """Get sentiment volume grouped by day for the past N days."""
    conn = sqlite3.connect(DB_PATH)
    since = (datetime.utcnow() - timedelta(days=days)).isoformat()
    rows = conn.execute(
        "SELECT date(created_at) as day, label, COUNT(*) as cnt "
        "FROM feedback WHERE created_at > ? GROUP BY day, label ORDER BY day",
        (since,)
    ).fetchall()
    conn.close()

    trend = {}
    for day, label, cnt in rows:
        if day not in trend:
            trend[day] = {"Positive": 0, "Negative": 0, "Neutral": 0}
        trend[day][label] = cnt
    return {"days": days, "trend": trend}

@app.get("/history")
def get_history(limit: int = 20, label: Optional[str] = None):
    """Fetch recent inference history."""
    conn = sqlite3.connect(DB_PATH)
    if label:
        rows = conn.execute(
            "SELECT raw_text, label, confidence, created_at FROM feedback "
            "WHERE label=? ORDER BY id DESC LIMIT ?", (label, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT raw_text, label, confidence, created_at FROM feedback "
            "ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    conn.close()
    return [{"text": r[0], "label": r[1], "confidence": r[2], "timestamp": r[3]} for r in rows]

@app.get("/stats")
def get_stats():
    """Aggregate statistics for dashboard KPIs."""
    conn = sqlite3.connect(DB_PATH)
    total   = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
    by_label = conn.execute(
        "SELECT label, COUNT(*) FROM feedback GROUP BY label"
    ).fetchall()
    avg_conf = conn.execute("SELECT AVG(confidence) FROM feedback").fetchone()[0]
    conn.close()
    counts = {row[0]: row[1] for row in by_label}
    return {
        "total": total,
        "positive": counts.get("Positive", 0),
        "negative": counts.get("Negative", 0),
        "neutral":  counts.get("Neutral", 0),
        "avg_confidence": round(avg_conf or 0, 2)
    }

@app.delete("/clear")
def clear_db():
    """Clear all feedback data (for testing)."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM feedback")
    conn.commit(); conn.close()
    return {"message": "Database cleared"}
