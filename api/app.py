from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline import RAGPipeline
app = FastAPI()

pipe = RAGPipeline.from_artifacts()

items = []


@app.get("/")
def root():
    return {"message": "Hello World"}


class AskRequest(BaseModel):
    question: str


@app.post("/api/ask")
def ask_question(req: AskRequest):
    print("Question received:", req.question)

    response = pipe.answer(req.question)

    return response
