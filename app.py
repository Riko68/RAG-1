import streamlit as st
import requests
import os
from typing import Dict, Any

def make_request(method: str, url: str, **kwargs) -> requests.Response:
    """Make an HTTP request with error handling."""
    try:
        response = requests.request(method, url, **kwargs)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {str(e)}")
        raise

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

st.title("RAG Q&A Demo")

# Sélection du rôle
role = st.radio("Select your role", ["user", "admin"], index=0)

# Entrée de la question
query = st.text_input("Ask a question:")
if st.button("Ask", key="ask_button"):
    try:
        resp = make_request("post", os.getenv("BACKEND_URL", "http://backend:8000") + "/ask", json={"query": query}, headers={"X-Role": role})
        if resp.status_code == 200:
            st.subheader("Answer:")
            st.write(resp.json()["answer"])
        else:
            st.error(f"Error {resp.status_code}: {resp.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Error: {str(e)}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")

# Admin section
if role == "admin":
    if st.button("Reindex documents", key="reindex_button"):
        try:
            resp = make_request("post", os.getenv("BACKEND_URL", "http://backend:8000") + "/admin/reindex", headers={"X-Role": "admin"}, timeout=5)
            if resp.status_code == 200:
                st.success("Reindex triggered.")
            else:
                st.error(f"Error {resp.status_code}: {resp.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Error triggering reindex: {str(e)}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

    # Liste des documents
    if st.button("Show indexed documents", key="documents_button"):
        try:
            resp = make_request("get", "http://backend:8000/documents", headers={"X-Role": "admin"}, timeout=5)
            docs = resp.json()["documents"]
            st.subheader("Indexed Documents:")
            if docs:
                for doc in docs:
                    st.markdown(f"- 📄 `{doc}`")
            else:
                st.info("No documents found.")
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching documents: {str(e)}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

    # Liste des documents
    if st.button("Show indexed documents"):
        try:
            resp = make_request("get", "http://backend:8000/documents", headers={"X-Role": "admin"}, timeout=5)
            docs = resp.json()["documents"]
            st.subheader("Indexed Documents:")
            if docs:
                for doc in docs:
                    st.markdown(f"- 📄 `{doc}`")
            else:
                st.info("No documents found.")
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching documents: {str(e)}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
