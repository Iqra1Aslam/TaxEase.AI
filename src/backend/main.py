import os
import re
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import google.generativeai as genai
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ==============================
# 1. Setup Google Gemini
# ==============================
genai.configure(api_key=os.getenv("AIzaSyB-SoFEpHmdQkE3LDcLKVblfRGA6PkywoI"))  # replace with os.getenv later

# ==============================
# 2. ChromaDB Setup
# ==============================
persist_dir = "./chroma_db"
client = chromadb.PersistentClient(path=persist_dir)

# 👇 specify cosine similarity for better matching
collection = client.get_or_create_collection(
    name="pdf_embeddings",
    metadata={"hnsw:space": "cosine"}
)

# ==============================
# 3. Embedding Model
# ==============================
model = SentenceTransformer("all-MiniLM-L6-v2")

# ==============================
# 4. Text Extraction & Chunking
# ==============================
def extract_pdf_text(file_path):
    reader = PdfReader(file_path)
    data = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            data.append({"page": i + 1, "content": text})
    return data

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into smaller overlapping chunks for better embeddings"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# ==============================
# 5. Only create embeddings if empty
# ==============================
if collection.count() == 0:
    print("⚡ No embeddings found in ChromaDB. Creating now...")

    pdf_path ="./data/instructions/1040-SE.pdf"
    pdf_data = extract_pdf_text(pdf_path)
    print(f"✅ Extracted {len(pdf_data)} pages from PDF")

    all_texts, all_metas, all_ids = [], [], []

    for item in pdf_data:
        chunks = chunk_text(item["content"])
        for idx, chunk in enumerate(chunks):
            all_texts.append(chunk)
            all_metas.append({"page": item["page"], "chunk": idx})
            all_ids.append(f"page_{item['page']}_chunk_{idx}")

    embeddings = model.encode(all_texts).tolist()
    print("✅ Created embeddings with chunking")

    collection.add(
        documents=all_texts,
        embeddings=embeddings,
        metadatas=all_metas,
        ids=all_ids
    )
    print("✅ Stored PDF embeddings in ChromaDB")
else:
    print(f"📂 Collection already has {collection.count()} embeddings. Skipping insert.")

# ==============================
# 6. Vector Search
# ==============================
def vector_search(query: str, top_k=5):
    print(f"\n🔍 Vector search for: {query}")
    query_vec = model.encode([query])[0].tolist()

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k
    )

    context = ""
    for i, (doc, meta, distance) in enumerate(
        zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
    ):
        # cosine distance → similarity = 1 - distance
        similarity = 1 - distance
        page = meta.get("page", "Unknown")
        snippet = doc[:400]
        context += (
            f"Result {i+1}\n"
            f"📄 Page: {page}, Chunk: {meta.get('chunk', '?')}\n"
            f"Similarity Score: {similarity:.4f}\n"
            f"{snippet}...\n---\n"
        )
    print("\n=== Retrieved Context ===\n", context)
    return context

# ==============================
# 7. Clean Response
# ==============================
def clean_response(response: str) -> str:
    response = re.sub(r"^As TaxEase\.AI.*?:", "", response, flags=re.IGNORECASE | re.DOTALL)
    response = response.replace("*", "").replace("#", "")
    response = re.sub(r"^\s*[-•]\s*", "", response, flags=re.MULTILINE)
    response = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", response)
    response = re.sub(r"\[.*?\]", "", response)
    response = re.sub(r"\((?!Form).*?\)", "", response)
    response = re.sub(r"\n{2,}", "\n\n", response)
    response = re.sub(r"[ \t]+", " ", response)
    return response.strip()

# ==============================
# 8. Gemini Integration
# ==============================
def ask_gemini(query: str):
    context = vector_search(query)
    prompt = f"""
    You are TaxEase.AI, a tax filing assistant.
    Answer the user’s query using the following context.
    If not found, answer from your own knowledge.

    USER QUERY: {query}

    CONTEXT:
    {context}
    """
    model_g = genai.GenerativeModel("gemini-2.5-pro")
    response = model_g.generate_content(prompt)
    return clean_response(response.text)

# ==============================
# 9. FastAPI Setup
# ==============================
app = FastAPI()

origins = ["http://localhost:3000", "https://your-ngrok-url.ngrok-free.app"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "TaxEase.AI backend running 🚀 with ChromaDB + Gemini"}

@app.get("/query/{query}")
async def get_answer(query: str):
    answer = ask_gemini(query)
    return {"response": answer}
