from fastapi import FastAPI
import snowflake.connector
from langchain_groq import ChatGroq
import os
import pandas as pd

app = FastAPI()

# -----------------------
# CONFIG (use env vars in real)
# -----------------------

SNOWFLAKE_CONFIG = {
    "user": os.getenv("SNOWFLAKE_USER"),
    "password": os.getenv("SNOWFLAKE_PASSWORD"),
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
    "database": os.getenv("SNOWFLAKE_DATABASE"),
    "schema": os.getenv("SNOWFLAKE_SCHEMA")
}

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

# -----------------------
# DB Connection
# -----------------------
def get_connection():
    return snowflake.connector.connect(**SNOWFLAKE_CONFIG)

# -----------------------
# SQL Validation
# -----------------------
def validate_sql(sql):
    sql_lower = sql.lower()

    if not sql_lower.startswith("select"):
        raise ValueError("Only SELECT allowed")

    if any(x in sql_lower for x in ["drop", "delete", "update", "insert"]):
        raise ValueError("Unsafe SQL detected")

    if "limit" not in sql_lower:
        sql += " LIMIT 50"

    return sql

# -----------------------
# Generate SQL
# -----------------------
def generate_sql(question):
    prompt = f"""
    Convert to Snowflake SQL.
    Only SELECT queries. No explanation.

    Question: {question}
    """

    response = llm.invoke(prompt)
    sql = response.content.strip().replace("```", "")
    return validate_sql(sql)

# -----------------------
# Run Query
# -----------------------
def run_query(sql):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(sql)
    df = cursor.fetch_pandas_all()
    cursor.close()
    conn.close()
    return df

# -----------------------
# Explain
# -----------------------
def explain(question, df):
    if df.empty:
        return "No data found"

    prompt = f"""
    Explain this result:

    Question: {question}
    Data:
    {df.head(10).to_string()}
    """

    return llm.invoke(prompt).content.strip()

# -----------------------
# API Endpoint
# -----------------------
@app.post("/query")
def query_api(payload: dict):
    question = payload.get("question")

    sql = generate_sql(question)
    df = run_query(sql)
    explanation = explain(question, df)

    return {
        "sql": sql,
        "data": df.to_dict(),
        "explanation": explanation
    }
