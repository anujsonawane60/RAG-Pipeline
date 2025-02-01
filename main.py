from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import PyPDF2
import io
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import cohere  # Cohere library
from enum import Enum
import re
from docx import Document

# Initialize FastAPI
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Cohere API Key
COHERE_API_KEY = "API_KEY"  # Replace with your Cohere API key

# Initialize Cohere client
cohere_client = cohere.Client(COHERE_API_KEY)

# Vector Database Setup
model = SentenceTransformer('all-MiniLM-L6-v2')
dimension = 384
index = faiss.IndexFlatL2(dimension)
text_chunks = []

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file))
    text = "".join([page.extract_text() for page in pdf_reader.pages])
    return text

def extract_text_from_docx(file):
    doc = Document(io.BytesIO(file))
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def add_to_vector_db(text):
    # Split text into sentences for better chunking
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < 500:  # Limit chunk size
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    for chunk in chunks:
        if chunk.strip():
            embedding = model.encode([chunk])[0]
            index.add(np.array([embedding]))
            text_chunks.append(chunk)
    
    print(f"Added {len(chunks)} chunks to the vector database.")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload-file/")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Check file type
        if file.content_type not in ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
            raise HTTPException(400, "Invalid file type. Please upload a PDF or DOCX file.")

        # Read file content
        file_content = await file.read()

        # Extract text based on file type
        if file.content_type == "application/pdf":
            text = extract_text_from_pdf(file_content)
        else:
            text = extract_text_from_docx(file_content)
        
        if not text.strip():
            raise HTTPException(400, "No text found in the document")
        
        print("Extracted Text:", text[:500])  # Log first 500 characters of extracted text
            
        add_to_vector_db(text)
        return {"status": "success", "message": "File processed successfully"}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.post("/ask")
async def ask_question(request: Request):
    try:
        data = await request.json()
        query = data.get("query")
        
        if not query:
            raise HTTPException(400, "Query parameter is required")

        # Generate query embedding
        query_embedding = model.encode([query])[0]

        # Search the FAISS index
        distances, indices = index.search(np.array([query_embedding]), 5)
        relevant_chunks = [text_chunks[idx] for idx in indices[0] if idx != -1]
        context = " ".join(relevant_chunks)

        if not context.strip():
            return JSONResponse(content={
                "answer": "No relevant information found in documents",
                "context": ""
            })

        # Generate answer using Cohere
        response = cohere_client.generate(
            prompt=f"Context:\n{context}\n\nQuestion: {query}\nAnswer:",
            max_tokens=500,
            temperature=0.7
        )
        answer = response.generations[0].text

        return JSONResponse(content={
            "status": "success",
            "answer": answer,
            "context": relevant_chunks,
            "provider": "Cohere"
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)