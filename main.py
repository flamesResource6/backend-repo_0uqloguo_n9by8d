import os
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from io import StringIO
import csv

# Try to import optional heavy deps; degrade gracefully
_SKLEARN_OK = True
_RAPIDFUZZ_OK = True
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
except Exception:  # ModuleNotFoundError or binary import error
    _SKLEARN_OK = False
    TfidfVectorizer = LinearSVC = Pipeline = LogisticRegression = None  # type: ignore

try:
    from rapidfuzz import process, fuzz
except Exception:
    _RAPIDFUZZ_OK = False
    process = fuzz = None  # type: ignore

try:
    import numpy as np
except Exception:
    np = None  # type: ignore

# Database helpers
from database import db, create_document, get_documents

app = FastAPI(title="MediBot — AI Medicine Chat")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Data models
# ----------------------------

INTENT_CLASSES = [
    "general",
    "uses",
    "how_to_take",
    "side_effects",
    "precautions",
    "interactions",
    "contraindications",
    "overdose_missed",
]

HIGH_RISK_PHRASES = [
    "severe chest pain",
    "anaphylaxis",
    "overdose",
    "fainting",
    "breathing trouble",
    "trouble breathing",
    "shortness of breath",
    "stroke",
    "heart attack",
]

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    json: Dict[str, Any]
    reply: str

class TrainIntentItem(BaseModel):
    text: str
    label: str

class FeedbackItem(BaseModel):
    message: str
    reply: str
    updown: int = Field(..., description="1 for upvote, -1 for downvote")
    edits: Optional[str] = None

# ----------------------------
# In-memory ML state (persist to DB for durability)
# ----------------------------

seed_intents: List[TrainIntentItem] = [
    TrainIntentItem(text="What is paracetamol?", label="general"),
    TrainIntentItem(text="Uses of ibuprofen", label="uses"),
    TrainIntentItem(text="How should I take amoxicillin?", label="how_to_take"),
    TrainIntentItem(text="Side effects of metformin", label="side_effects"),
    TrainIntentItem(text="What precautions for warfarin?", label="precautions"),
    TrainIntentItem(text="Does aspirin interact with warfarin?", label="interactions"),
    TrainIntentItem(text="Who should avoid isotretinoin?", label="contraindications"),
    TrainIntentItem(text="Missed dose of insulin, what to do?", label="overdose_missed"),
]

intent_model = None  # type: ignore
meds_index: List[Dict[str, str]] = []
reward_model = None  # type: ignore

# ----------------------------
# Utility functions
# ----------------------------

def load_intent_training_data() -> List[TrainIntentItem]:
    items = [TrainIntentItem(**d) for d in get_documents("intent", {})] if db else []
    if not items:
        items = seed_intents
    return items


def train_intent_model():
    global intent_model
    data = load_intent_training_data()
    X = [it.text for it in data]
    y = [it.label for it in data]
    for lbl in y:
        if lbl not in INTENT_CLASSES:
            raise HTTPException(status_code=400, detail=f"Unknown label: {lbl}")

    if _SKLEARN_OK:
        intent_model = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("clf", LinearSVC()),
        ])
        intent_model.fit(X, y)
    else:
        # Fallback: simple keyword rules
        intent_model = "rules"  # type: ignore


def predict_intent(text: str) -> str:
    global intent_model
    if intent_model is None:
        train_intent_model()
    if intent_model == "rules":
        t = text.lower()
        if any(k in t for k in ["use", "treat", "for "]):
            return "uses"
        if any(k in t for k in ["take", "dose", "dosage", "how to"]):
            return "how_to_take"
        if any(k in t for k in ["side effect", "adverse", "nausea", "dizziness"]):
            return "side_effects"
        if any(k in t for k in ["precaution", "careful", "warning"]):
            return "precautions"
        if any(k in t for k in ["interact", "interaction", "with "]):
            return "interactions"
        if any(k in t for k in ["avoid", "contraindicat", "not use"]):
            return "contraindications"
        if any(k in t for k in ["overdose", "missed dose", "skip dose"]):
            return "overdose_missed"
        return "general"
    else:
        return intent_model.predict([text])[0]


def update_meds_index_from_csv(csv_text: str):
    global meds_index
    meds_index = []
    reader = csv.DictReader(StringIO(csv_text))
    for row in reader:
        name = (row.get("name") or "").strip()
        typ = (row.get("type") or "generic").strip().lower()
        if not name:
            continue
        if typ not in {"generic", "brand"}:
            typ = "generic"
        meds_index.append({"name": name, "type": typ})


def resolve_medicine(query: str) -> Optional[str]:
    if not meds_index:
        docs = get_documents("med", {}) if db else []
        if docs:
            combined = "name,type\n" + "\n".join([f"{d.get('name','')},{d.get('type','generic')}" for d in docs])
            update_meds_index_from_csv(combined)
    if not meds_index:
        return None
    names = [m["name"] for m in meds_index]
    if _RAPIDFUZZ_OK:
        match, score, _ = process.extractOne(query, names, scorer=fuzz.token_sort_ratio)
        return match if score >= 70 else None
    else:
        # Simple regex-like substring search fallback
        q = query.lower()
        best = None
        best_len = 0
        for n in names:
            if n.lower() in q or q in n.lower():
                if len(n) > best_len:
                    best = n
                    best_len = len(n)
        return best


def safety_flag(text: str) -> bool:
    t = text.lower()
    return any(phrase in t for phrase in HIGH_RISK_PHRASES)


def response_features(d: Dict[str, Any]):
    bullets = d.get("bullets", [])
    text = " ".join([" ".join(b.get("points", [])) if isinstance(b, dict) else str(b) for b in bullets])
    feats = [
        len(text),
        len(bullets),
        1 if d.get("urgent") else 0,
        text.lower().count("avoid"),
        text.lower().count("warning"),
        text.lower().count("side effect"),
    ]
    if np is not None:
        return np.array(feats).reshape(1, -1)
    return [[f for f in feats]]


def dpo_rerank(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not candidates:
        return {"bullets": ["Sorry, I couldn't draft a reply."], "urgent": False}
    if reward_model is None or np is None or not hasattr(reward_model, "predict_proba"):
        return candidates[0]
    scores = [reward_model.predict_proba(response_features(c))[0, 1] for c in candidates]
    return candidates[int(np.argmax(scores))]


# ----------------------------
# Draft generation
# ----------------------------

def generate_draft(message: str, med: Optional[str], intent: str) -> Dict[str, Any]:
    urgent = safety_flag(message)

    bullets: List[Dict[str, Any]] = []
    name = med or "this medicine"

    def add(title: str, body: List[str]):
        bullets.append({"title": title, "points": body})

    if intent == "general":
        add("What it is", [f"{name}: commonly used medicine.", "Belongs to a known class."])
        add("How it works", ["Acts on specific pathways to reduce symptoms."])
        add("Key safety notes", ["Avoid if allergic", "Check interactions if on other meds"])    
    elif intent == "uses":
        add("Common uses", [f"Used for typical conditions related to {name}."])
        add("Sometimes used for", ["Doctor may use off-label in some cases."])
    elif intent == "how_to_take":
        add("General how to take", ["Follow the exact label.", "With or without food unless told otherwise."])
        if not urgent:
            add("Dosing guidance (general)", ["Do not exceed label dose.", "Measure liquids accurately."])
        add("Storage", ["Keep in a cool, dry place.", "Away from children."])
    elif intent == "side_effects":
        add("Common", ["Nausea, headache, dizziness—often mild."])
        add("Serious—seek urgent care", ["Allergic reaction, severe chest pain, trouble breathing."])
    elif intent == "precautions":
        add("Before using", ["Tell your doctor about allergies, pregnancy, liver/kidney issues."])
        add("Who needs extra caution", ["Older adults, multiple medicines, chronic conditions."])
    elif intent == "interactions":
        add("Common interactions", ["Other medicines that affect the same pathway."])
        add("Avoid combining with", ["Alcohol or duplicate therapies unless advised."])
    elif intent == "contraindications":
        add("Do not use if", ["Severe allergy to this medicine or similar."])
        add("Talk to a doctor if", ["Pregnant/breastfeeding or serious conditions."])
    elif intent == "overdose_missed":
        add("Overdose basics", ["If overdose suspected: call emergency services now."])
        add("Missed dose basics", ["Take when remembered unless close to next—do not double."])

    draft = {"medicine": med, "intent": intent, "urgent": urgent, "bullets": bullets}

    candidates = [draft]
    alt = dict(draft)
    alt["bullets"] = draft["bullets"] + [{"title": "General tips", "points": ["Read the leaflet", "Keep a list of your meds"]}]
    candidates.append(alt)
    return dpo_rerank(candidates)


def render_reply(d: Dict[str, Any]) -> str:
    lines: List[str] = []
    if d.get("urgent"):
        lines.append("🚨 Urgent safety alert: If you have severe symptoms (like trouble breathing, chest pain, fainting, anaphylaxis, or overdose), seek emergency care immediately.")
    for sec in d.get("bullets", []):
        title = sec.get("title")
        points = sec.get("points", [])
        if title:
            lines.append(f"• {title}:")
        for p in points:
            lines.append(f"  - {p}")
    lines.append("This is general information only. Please consult a doctor before taking or stopping any medicine.")
    return "\n".join(lines)


# ----------------------------
# Endpoints
# ----------------------------

@app.get("/")
def root():
    return {
        "app": "MediBot",
        "status": "ok",
        "sklearn": _SKLEARN_OK,
        "rapidfuzz": _RAPIDFUZZ_OK,
    }

@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["connection_status"] = "Connected"
            response["collections"] = db.list_collection_names()[:10]
        else:
            response["database"] = "❌ Not Configured"
    except Exception as e:
        response["database"] = f"⚠️ Error: {str(e)[:80]}"
    return response


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Empty message")

    med = resolve_medicine(message) if any(c.isalpha() for c in message) else None

    intent = predict_intent(message)

    draft = generate_draft(message, med, intent)

    reply = render_reply(draft)

    if db is not None:
        try:
            create_document("chat", {"message": message, "reply": reply, "json": draft})
        except Exception:
            pass

    return ChatResponse(json=draft, reply=reply)


@app.post("/upload/meds.csv")
async def upload_meds(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Please upload a CSV file")
    text = (await file.read()).decode("utf-8", errors="ignore")
    update_meds_index_from_csv(text)

    if db is not None:
        try:
            db["med"].delete_many({})
            reader = csv.DictReader(StringIO(text))
            docs = []
            for row in reader:
                name = (row.get("name") or "").strip()
                typ = (row.get("type") or "generic").strip().lower()
                if name:
                    docs.append({"name": name, "type": typ})
            if docs:
                db["med"].insert_many(docs)
        except Exception:
            pass
    return {"status": "ok", "count": len(meds_index)}


@app.post("/train/intent")
async def train_intent(items: List[TrainIntentItem]):
    if db is not None:
        for it in items:
            if it.label not in INTENT_CLASSES:
                raise HTTPException(status_code=400, detail=f"Unknown label: {it.label}")
            create_document("intent", it.model_dump())
    train_intent_model()
    return {"status": "retrained", "classes": INTENT_CLASSES, "sklearn": _SKLEARN_OK}


@app.post("/feedback")
async def feedback(item: FeedbackItem):
    if db is not None:
        create_document("feedback", item.model_dump())
    return {"status": "recorded"}


@app.post("/train/style")
async def train_style():
    global reward_model
    fb = get_documents("feedback", {}) if db else []
    if not fb or not _SKLEARN_OK or np is None:
        reward_model = None
        return {"status": "no_data"}

    X = []
    y = []
    for f in fb:
        draft = f.get("json") or {}
        if not draft:
            draft = {"bullets": [{"title": None, "points": f.get("reply", "").split("\n")}], "urgent": False}
        X.append(response_features(draft).ravel())
        y.append(1 if int(f.get("updown", 0)) > 0 else 0)
    X = np.vstack(X)
    reward_model = LogisticRegression(max_iter=1000)
    reward_model.fit(X, y)
    return {"status": "trained", "samples": len(y)}


@app.post("/train/lora")
async def train_lora():
    return {"status": "skipped", "reason": "No GPU available; only style/intent models are trained."}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
