import streamlit as st
import snowflake.connector
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import pandas as pd

# ✅ RAG imports
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

load_dotenv()

# -----------------------
# Snowflake Connection
# -----------------------
def get_connection():
    return snowflake.connector.connect(
        user=st.secrets["SNOWFLAKE_USER"],
        password=st.secrets["SNOWFLAKE_PASSWORD"],
        account=st.secrets["SNOWFLAKE_ACCOUNT"],
        warehouse=st.secrets["SNOWFLAKE_WAREHOUSE"],
        database=st.secrets["SNOWFLAKE_DATABASE"],
        schema=st.secrets["SNOWFLAKE_SCHEMA"]
    )

# -----------------------
# LLM (Groq)
# -----------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=st.secrets["GROQ_API_KEY"]
)

# -----------------------
# Multi Schema
# -----------------------
SCHEMAS = {
    "GOOGLE REPORT DATA": {
        "table": "GOOG_GLOBAL_MOBILITY_REPORT",
        "schema": """
        Table: GOOG_GLOBAL_MOBILITY_REPORT
        Columns:
        - COUNTRY_REGION
        - DATE
        - GROCERY_AND_PHARMACY_CHANGE_PERC
        - PARKS_CHANGE_PERC
        - RESIDENTIAL_CHANGE_PERC
        """
    },
    "NYC HEALTH TESTS DATA": {
        "table": "NYC_HEALTH_TESTS",
        "schema": """
        Table: NYC_HEALTH_TESTS
        Columns:
        - COUNTRY_REGION
        - COVID_CASE_COUNT
        - DATE
        - TOTAL_COVID_TESTS
        - PERCENT_POSITIVE
        """
    }
}

# -----------------------
# ✅ RAG Setup
# -----------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

schema_texts = [v["schema"] for v in SCHEMAS.values()]
schema_keys = list(SCHEMAS.keys())

embeddings = model.encode(schema_texts)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

def get_relevant_schema(question):
    q_embedding = model.encode([question])
    D, I = index.search(np.array(q_embedding), k=1)
    key = schema_keys[I[0][0]]
    return SCHEMAS[key]["schema"], SCHEMAS[key]["table"], key

# -----------------------
# ✅ SQL Validation
# -----------------------
def validate_sql(sql):
    sql_lower = sql.lower()

    if not sql_lower.startswith("select"):
        raise ValueError("Only SELECT queries allowed")

    blocked = ["drop", "delete", "truncate", "update", "insert"]
    if any(word in sql_lower for word in blocked):
        raise ValueError("Dangerous SQL detected")

    if "limit" not in sql_lower:
        sql += " LIMIT 50"

    return sql

# -----------------------
# Generate SQL
# -----------------------
def generate_sql(question, schema_text, table_name):
    prompt = f"""
    You are a Snowflake SQL expert.

    Rules:
    - Only SELECT queries
    - Use correct columns
    - Always use table: {table_name}
    - Add LIMIT 50
    - No explanation
    - No markdown

    Schema:
    {schema_text}

    Question:
    {question}
    """

    response = llm.invoke(prompt)
    sql = response.content.strip()
    sql = sql.replace("```sql", "").replace("```", "").strip()

    return validate_sql(sql)

# -----------------------
# ✅ SQL AUTO FIX LOOP
# -----------------------
def run_query_with_fix(question, sql, max_retries=2):
    for attempt in range(max_retries):
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute(sql)
            df = cursor.fetch_pandas_all()
            cursor.close()
            conn.close()
            return df, sql

        except Exception as e:
            error_msg = str(e)

            fix_prompt = f"""
            Fix this Snowflake SQL.

            SQL:
            {sql}

            Error:
            {error_msg}

            Rules:
            - Return only corrected SQL
            - No explanation
            """

            response = llm.invoke(fix_prompt)
            sql = response.content.strip()
            sql = sql.replace("```sql", "").replace("```", "").strip()
            sql = validate_sql(sql)

    raise Exception("SQL failed after retries")

# -----------------------
# Explain Result
# -----------------------
def explain_result(question, df):
    if df.empty:
        return "No data found."

    prompt = f"""
    Explain this data simply:

    Question:
    {question}

    Data:
    {df.head(10).to_string()}
    """

    response = llm.invoke(prompt)
    return response.content.strip()

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Snowflake AI Chat", layout="wide")

st.title("💬 AI Snowflake Chatbot (Advanced)")
st.caption("🚀 Now with RAG + Auto-Fix + Validation")

if "history" not in st.session_state:
    st.session_state.history = []

question = st.chat_input("Ask your question...")

if question:
    with st.spinner("Thinking..."):
        try:
            # ✅ RAG selection
            schema_text, table_name, dataset = get_relevant_schema(question)

            sql = generate_sql(question, schema_text, table_name)

            # ✅ Auto fix execution
            df, sql = run_query_with_fix(question, sql)

            explanation = explain_result(question, df)

            st.session_state.history.append({
                "role": "user",
                "content": question
            })

            st.session_state.history.append({
                "role": "assistant",
                "sql": sql,
                "result": df,
                "explanation": explanation,
                "dataset": dataset
            })

        except Exception as e:
            st.error(f"Error: {str(e)}")

# -----------------------
# Chat Display
# -----------------------
for chat in st.session_state.history:

    if chat["role"] == "user":
        with st.chat_message("user"):
            st.write(chat["content"])

    elif chat["role"] == "assistant":
        with st.chat_message("assistant"):

            st.markdown(f"📊 **Dataset:** {chat['dataset']}")

            st.markdown("**SQL:**")
            st.code(chat["sql"], language="sql")

            st.dataframe(chat["result"])

            if not chat["result"].empty:
                csv = chat["result"].to_csv(index=False)
                #st.download_button("⬇️ Download CSV", csv)
                st.download_button(
                    "⬇️ Download CSV",
                    data=csv,
                    file_name=f"{chat['dataset']}.csv",
                    mime="text/csv",
                    key=f"download_{i}"
                )

            st.markdown("**Explanation:**")
            st.write(chat["explanation"])

            if not chat["result"].empty:
                df = chat["result"].dropna()
                numeric_cols = df.select_dtypes(include=['number']).columns

                if len(numeric_cols) > 0:
                    st.bar_chart(df[numeric_cols])
                    
