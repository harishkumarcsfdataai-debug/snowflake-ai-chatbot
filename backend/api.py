from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq

print("🔥 THIS API FILE IS RUNNING")
# -----------------------
# Load env
# -----------------------
load_dotenv()

app = FastAPI()

# -----------------------
# LLM
# -----------------------
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama3-70b-8192"
)

# -----------------------
# Request Model (IMPORTANT)
# -----------------------
class QueryRequest(BaseModel):
    question: str

# -----------------------
# API
# -----------------------
@app.post("/query")
def query_api(req: QueryRequest):
    response = llm.invoke(req.question)

    return {
        "answer": response.content
    }
