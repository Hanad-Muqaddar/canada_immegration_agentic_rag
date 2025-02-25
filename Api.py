from fastapi import FastAPI
from pydantic import BaseModel
# from Retrival import RAGChatbot
import uvicorn
from AgenticRag import return_output

app = FastAPI()

# Initialize chatbot once at startup
# chatbot = RAGChatbot()

class QuestionRequest(BaseModel):
    question: str
    id : str

class AnswerResponse(BaseModel):
    answer: str

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    # answer = chatbot.ask_question(request.question)
    response = return_output(request.question, request.id)
    return {"answer": response}


if __name__ == "__main__":
    uvicorn.run("Api:app", host="0.0.0.0", port=8001, reload=True)



# OPENAI_API_KEY = "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"
