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
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

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
    result = classifier(text)[0]
    label = result['label']
    confidence = round(result['score'] * 100, 2)
    return label, confidence, text

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
