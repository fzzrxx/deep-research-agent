from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from graph import graph

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ResearchRequest(BaseModel):
    query: str

@app.get("/")
def home():
    return {"message": "Deep Research Agent is running"}

@app.post("/research")
def research(request: ResearchRequest):
    result = graph.invoke({
        "messages": [{"role": "user", "content": request.query}],
        "report": None
    })
    report = result["report"]
    return {
        "title": report.title,
        "summary": report.summary,
        "key_findings": report.key_findings,
        "citations": [{"url": c.source_url, "point": c.key_point} for c in report.citations],
        "follow_up_questions": report.follow_up_questions
    }