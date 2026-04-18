import streamlit as st
import requests
import pandas as pd

# -----------------------
# CONFIG
# -----------------------
API_URL = "http://127.0.0.1:8000/query"

st.set_page_config(page_title="Snowflake AI Chat", layout="wide")

st.title("💬 AI Snowflake Chatbot")
st.caption("Streamlit UI → FastAPI → Snowflake + Groq")

# -----------------------
# SESSION STATE
# -----------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------
# INPUT
# -----------------------
question = st.chat_input("Ask your question...")

# -----------------------
# CALL FASTAPI
# -----------------------
def call_api(question):
    response = requests.post(
        API_URL,
        json={"question": question},
        timeout=60
    )
    return response.json()

# -----------------------
# USER INPUT HANDLING
# -----------------------
if question:
    with st.spinner("Thinking..."):
        try:
            result = call_api(question)

            if "error" in result:
                st.error(result["error"])
            else:
                st.session_state.history.append({
                    "role": "user",
                    "content": question
                })

                st.session_state.history.append({
                    "role": "assistant",
                    "sql": result["sql"],
                    "data": result["data"],
                    "explanation": result["explanation"]
                })

        except Exception as e:
            st.error(str(e))

# -----------------------
# CHAT DISPLAY
# -----------------------
for chat in st.session_state.history:

    if chat["role"] == "user":
        with st.chat_message("user"):
            st.write(chat["content"])

    elif chat["role"] == "assistant":
        with st.chat_message("assistant"):

            st.markdown("**SQL Generated:**")
            st.code(chat["sql"], language="sql")

            df = pd.DataFrame(chat["data"])
            st.dataframe(df)

            # CSV download
            if not df.empty:
                csv = df.to_csv(index=False)

                st.download_button(
                    "⬇️ Download CSV",
                    data=csv,
                    file_name="result.csv",
                    mime="text/csv"
                )

            st.markdown("**Explanation:**")
            st.write(chat["explanation"])

            # Chart
            if not df.empty:
                numeric_cols = df.select_dtypes(include=["number"]).columns
                if len(numeric_cols) > 0:
                    st.bar_chart(df[numeric_cols])
