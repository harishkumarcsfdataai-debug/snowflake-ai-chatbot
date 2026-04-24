from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq

# -----------------------
# Load environment
# -----------------------
load_dotenv()

# -----------------------
# Initialize FastAPI
# -----------------------
app = FastAPI()

# -----------------------
# Initialize LLM (Groq)
# -----------------------
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama3-70b-8192"
)

# -----------------------
# Request schema
# -----------------------
class QueryRequest(BaseModel):
    question: str

# -----------------------
# API Endpoint
# -----------------------
@app.post("/query")
def query_api(req: QueryRequest):
    try:
        user_question = req.question

        # Call LLM
        response = llm.invoke(user_question)

        return {
            "answer": response.content
        }

    except Exception as e:
        return {
            "error": str(e)
        }
