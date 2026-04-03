import os
import snowflake.connector
import pandas as pd
from dotenv import load_dotenv
#from langchain.chat_models import ChatOpenAI
#from langchain_community.chat_models import ChatOpenAI
#from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama

load_dotenv()

# -----------------------
# Snowflake Connection
# -----------------------
def get_connection():
    return snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA")
    )

# -----------------------
# LLM Setup
# -----------------------
#llm = ChatOpenAI(
#    temperature=0,
#    openai_api_key=os.getenv("OPENAI_API_KEY")
#)
llm = ChatOllama(
    model="llama3"
)

# -----------------------
# Schema (EDIT THIS)
# -----------------------
SCHEMA = """
Table: catalog_sales
Columns:
- cs_order_number (int)
- cs_sales_price (float)
- cs_quantity (int)
- cs_ship_date_sk (int)
"""

# -----------------------
# Generate SQL
# -----------------------
def generate_sql(question):
    prompt = f"""
    You are a Snowflake SQL expert.

    Convert the question into SQL.

    Rules:
    - Only SELECT queries
    - Use correct column names
    - Add LIMIT 50
    - No explanation

    Schema:
    {SCHEMA}

    Question:
    {question}
    """

    response = llm.invoke(prompt)
    sql = response.content.strip()

    # 🔥 CLEAN THE OUTPUT
    sql = sql.replace("```sql", "").replace("```", "").strip()

    return sql
# -----------------------
# Execute SQL
# -----------------------
def execute_sql(query):
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(query)
        return cursor.fetch_pandas_all()
    finally:
        cursor.close()

        conn.close()

# -----------------------
# MAIN TEST
# -----------------------
if __name__ == "__main__":
    question = input("Ask your question: ")

    sql = generate_sql(question)
    print("\nGenerated SQL:\n", sql)

    try:
        df = execute_sql(sql)
        print("\nResult:\n", df.head())
    except Exception as e:
        print("Error:", e)