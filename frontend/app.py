import streamlit as st
import requests

st.title("RAG Q&A Demo")

role = st.radio("Choose your role:", ["user", "admin"])

query = st.text_input("Ask a question:")
if st.button("Ask"):
    resp = requests.post("http://backend:8000/ask", params={"query": query}, headers={"X-Role": role})
    st.write(resp.json())

if role == "admin":
    if st.button("Trigger reindex"):
        resp = requests.post("http://backend:8000/admin/reindex", headers={"X-Role": "admin"})
        st.write(resp.json())
