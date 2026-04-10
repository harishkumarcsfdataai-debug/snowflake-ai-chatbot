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
# Multi Schema Support
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
        - FIPS
        - ISO3166_1
        - ISO3166_2
        - LAST_REPORTED_DATE
        - LAST_UPDATED_DATE
        - MODIFIED_ZCTA
        - PERCENT_POSITIVE
        - TOTAL_COVID_TESTS
        """
    }
}

# -----------------------
# Generate SQL
# -----------------------
def generate_sql(question, schema_text, table_name):
    prompt = f"""
    You are a Snowflake SQL expert.

    Convert the question into SQL.

    Rules:
    - Only SELECT queries
    - Use correct column names
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
# Explain Result
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
st.caption("🚀 AI-powered Snowflake Analytics Assistant")

# Dataset selector
dataset = st.selectbox(
    "📊 Choose Dataset",
    list(SCHEMAS.keys())
)

# Session state
if "history" not in st.session_state:
    st.session_state.history = []

# Chat input
question = st.chat_input("Ask your question...")

# Handle input
if question:
    with st.spinner("Thinking..."):
        try:
            selected_schema = SCHEMAS[dataset]["schema"]
            table_name = SCHEMAS[dataset]["table"]

            sql = generate_sql(question, selected_schema, table_name)
            df = run_query(sql)
            explanation = explain_result(question, df)

            # Save user message
            st.session_state.history.append({
                "role": "user",
                "content": question
            })

            # Save assistant response
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

    # User message
    if chat["role"] == "user":
        with st.chat_message("user"):
            st.write(chat["content"])

    # Assistant message
    elif chat["role"] == "assistant":
        with st.chat_message("assistant"):

            st.markdown(f"📊 **Dataset:** {chat['dataset']}")

            st.markdown("**Generated SQL:**")
            st.code(chat["sql"], language="sql")

            st.markdown("**Result:**")
            st.dataframe(chat["result"])

            # Download button
            if not chat["result"].empty:
                csv = chat["result"].to_csv(index=False)
                st.download_button(
                    "⬇️ Download CSV",
                    data=csv,
                    file_name=f"{chat['dataset']}.csv",
                    mime="text/csv"
                )

            # Explanation
            st.markdown("**Explanation:**")
            st.write(chat["explanation"])

            # Smart chart
            if not chat["result"].empty:
                df = chat["result"].dropna()
                numeric_cols = df.select_dtypes(include=['number']).columns

                if len(numeric_cols) > 0:
                    st.markdown("**Chart:**")
                    st.bar_chart(df[numeric_cols])