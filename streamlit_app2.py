import streamlit as st
import snowflake.connector
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import pandas as pd

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
# Schema
# -----------------------
SCHEMA = """
Table: GOOG_GLOBAL_MOBILITY_REPORT
Columns:
- COUNTRY_REGION
- DATE
- GROCERY_AND_PHARMACY_CHANGE_PERC
- PARKS_CHANGE_PERC
- RESIDENTIAL_CHANGE_PERC
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
    - No markdown (no ```)

    Schema:
    {SCHEMA}

    Question:
    {question}
    """

    response = llm.invoke(prompt)
    sql = response.content.strip()
    sql = sql.replace("```sql", "").replace("```", "").strip()

    return sql

# -----------------------
# Execute SQL
# -----------------------
def run_query(query):
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(query)
        return cursor.fetch_pandas_all()
    finally:
        cursor.close()
        conn.close()

# -----------------------
# Explain Result ✅ FIXED POSITION
# -----------------------
def explain_result(question, df):
    if df.empty:
        return "No data found for this query."

    prompt = f"""
    You are a data analyst.

    User question:
    {question}

    Data result:
    {df.head(10).to_string()}

    Explain the result in simple English.
    Highlight key insights briefly.
    """

    response = llm.invoke(prompt)
    return response.content.strip()

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Snowflake AI Chat", layout="wide")

st.title("💬 Chat with Snowflake Data (AI Powered)")

if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_input("Ask your question:")

if st.button("Submit") and question:
    with st.spinner("Thinking..."):
        try:
            sql = generate_sql(question)
            df = run_query(sql)
            explanation = explain_result(question, df)

            st.session_state.history.append({
                "question": question,
                "sql": sql,
                "result": df,
                "explanation": explanation
            })

        except Exception as e:
            st.error(f"Error: {str(e)}")

# -----------------------
# Display Chat History
# -----------------------
for chat in reversed(st.session_state.history):
    st.markdown(f"### 🧑‍💼 You: {chat['question']}")

    st.markdown("**Generated SQL:**")
    st.code(chat["sql"], language="sql")

    st.markdown("**Result:**")
    st.dataframe(chat["result"])

    # ✅ Explanation
    if "explanation" in chat:
        st.markdown("**Explanation:**")
        st.write(chat["explanation"])

    # ✅ Smart Chart (NULL-safe)
    if not chat["result"].empty:
        df = chat["result"].dropna()

        numeric_cols = df.select_dtypes(include=['number']).columns

        if len(numeric_cols) > 0 and not df.empty:
            st.markdown("**Chart:**")
            st.bar_chart(df[numeric_cols])