from fastapi import FastAPI, Depends, Header, HTTPException

app = FastAPI()

def get_role(x_role: str = Header(...)):
    if x_role not in ["admin", "user"]:
        raise HTTPException(status_code=401, detail="Invalid role")
    return x_role

@app.get("/health")
def health():
    return {"status": "ok"}

""" @app.post("/ask")
def ask_question(query: str, role: str = Depends(get_role)):
    # Call Qdrant + Ollama, return answer (stub for now)
    return {"answer": "This is a placeholder answer.", "sources": []} """

@app.post("/admin/reindex")
def reindex(role: str = Depends(get_role)):
    if role != "admin":
        raise HTTPException(status_code=403, detail="Admins only")
    # Call embedding worker or trigger reindex (stub for now)
    return {"message": "Reindex triggered."}

@app.get("/documents")
def list_documents():
    import os
    docs_path = "documents"
    files = os.listdir(docs_path)
    return {"documents": files}

@app.post("/ask")
def query_endpoint(query: QueryRequest):
    response = rag_chain.run(query.query)
    return {"answer": response}


