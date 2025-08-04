import streamlit as st
import requests

st.title("RAG Q&A Demo")

# Sélection du rôle
role = st.radio("Choose your role:", ["user", "admin"])

# Entrée de la question
query = st.text_input("Ask a question:")
if st.button("Ask"):
    # Requête POST avec body JSON et header X-Role
    resp = requests.post(
        "http://backend:8000/ask",
        json={"query": query},
        headers={"X-Role": role}
    )
    if resp.status_code == 200:
        st.subheader("Answer:")
        st.write(resp.json()["answer"])
    else:
        if resp.status_code == 200:
            st.subheader("Answer:")
            st.write(resp.json()["answer"])
        else:
            st.error(f"Error {resp.status_code}: {resp.text}")

# Section admin
if role == "admin":
    if st.button("Trigger reindex"):
        # Wait for backend to be ready
        if not wait_for_backend():
            st.error("Backend service not responding. Please try again later.")
            st.stop()

        try:
            resp = make_request("post", "http://backend:8000/admin/reindex", headers={"X-Role": "admin"}, timeout=5)
            if resp.status_code == 200:
                st.success("Reindex triggered.")
            else:
                st.error(f"Error {resp.status_code}: {resp.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Error triggering reindex: {str(e)}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

    # Liste des documents
    if st.button("Show indexed documents"):
        # Wait for backend to be ready
        if not wait_for_backend():
            st.error("Backend service not responding. Please try again later.")
            st.stop()

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
