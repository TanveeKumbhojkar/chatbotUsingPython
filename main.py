from dotenv import load_dotenv
load_dotenv()  

from langchain_community.utilities.sql_database import SQLDatabase
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, create_sql_query_chain
import os
import streamlit as st


def load_db():
    db = SQLDatabase.from_uri("mysql+pymysql://root:root@192.168.29.45:3306/real_estate")
    return db

def chain_create(db):
    llm = ChatGroq(model="mixtral-8x7b-32768", api_key=os.getenv("GROQ_API_KEY"))
    chain = create_sql_query_chain(llm, db)
    return chain

def sql_infer(db, chain, user_question):
    sql_query = chain.invoke({"question": user_question})
    result = db.run(sql_query)

    st.code(sql_query)
    st.write(result)

    answer_prompt = PromptTemplate.from_template(
        """Given the following user question, SQL query, and SQL result, generate a reply.

        Question: {question}
        SQL Query: {query}
        SQL Result: {result}
        Answer:"""
    )

    llm_model = ChatGroq(model="mixtral-8x7b-32768", api_key=os.getenv("GROQ_API_KEY"))
    llm = LLMChain(llm=llm_model, prompt=answer_prompt)
    ans = llm(inputs={"question": user_question, "query": sql_query, "result": result})
    return ans["text"]

def main():
    st.set_page_config(page_icon="🏠", layout="wide", page_title="Real Estate DB QA")
    st.title("Real Estate Database Question Answering App 🏡")

    db = load_db()
    chain = chain_create(db)

    st.subheader("Table Names")
    st.code(db.get_usable_table_names())

    st.subheader("Table Schemas")
    st.code(db.get_table_info())

    question = st.text_input("Ask a question about your real estate data")
    if st.button("Get Answer"):
        if question:
            try:
                answer = sql_infer(db, chain, question)
                st.subheader("Answer")
                st.write(answer)
            except Exception as e:
                st.error(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
