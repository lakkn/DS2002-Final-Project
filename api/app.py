from fastapi import FastAPI
from pydantic import BaseModel
# import getResponse from '\rag_pipeline.rag'
app = FastAPI()

items = []


@app.get("/")
def root():
    return {"message": "Hello World"}


class AskRequest(BaseModel):
    question: str


@app.post("/api/ask")
def ask_question(req: AskRequest):
    print("Question received:", req.question)
    # reponse = getResponse(req.question)
    # data cleaning?
    return {
        "answer": "babababa",
        "sources": [
            {"id": "filename.pdf", "page": 7, "snippet": "This is a dummy snippet from filename.pdf, page 7."},
        ]
    }
