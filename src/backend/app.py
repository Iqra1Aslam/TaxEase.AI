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
genai.configure(api_key="AIzaSyB-SoFEpHmdQkE3LDcLKVblfRGA6PkywoI")  # replace with os.getenv later

# ==============================
# 2. ChromaDB Setup
# ==============================
persist_dir = r"D:\tax\TaxEase.AI-Vertex-AI-Agent\src\backend\chroma_db"
client = chromadb.PersistentClient(path=persist_dir)

# client.delete_collection("pdf_embeddings_8027")
# print("🗑️ Old collection deleted")

# 👇 specify cosine similarity for better matching
collection = client.get_or_create_collection(
    name="pdf_embeddings_8027",
    metadata={"hnsw:space": "cosine"}
)
# print("✅ New empty collection created")
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

# def chunk_text(text, chunk_size=500, overlap=50):
#     """Split text into smaller overlapping chunks for better embeddings"""
#     words = text.split()
#     chunks = []
#     for i in range(0, len(words), chunk_size - overlap):
#         chunk = " ".join(words[i:i + chunk_size])
#         chunks.append(chunk)
#     return chunks
# def chunk_text(text, chunk_size=400, overlap=50):
#     """
#     Smarter chunking for IRS instructions:
#     - Split by paragraphs/lines first
#     - Merge smaller parts
#     - Apply max size with overlap
#     """
#     # 1. Split into paragraphs or lines
#     paragraphs = re.split(r'\n{2,}|\n', text)  # handles line/paragraph breaks
#     paragraphs = [p.strip() for p in paragraphs if p.strip()]

#     chunks = []
#     current_chunk = []

#     for para in paragraphs:
#         words = para.split()
        
#         # If adding this paragraph exceeds chunk size → save current chunk
#         if sum(len(p.split()) for p in current_chunk) + len(words) > chunk_size:
#             if current_chunk:
#                 chunks.append(" ".join(current_chunk))
#                 # overlap: keep last `overlap` words from previous chunk
#                 if overlap > 0:
#                     overlap_words = " ".join(current_chunk[-overlap:])
#                     current_chunk = [overlap_words]
#                 else:
#                     current_chunk = []
#         current_chunk.append(para)

#     # Add remaining chunk
#     if current_chunk:
#         chunks.append(" ".join(current_chunk))

#     return chunks


def chunk_text(text, chunk_size=500, overlap=50):
    """
    Chunk IRS instruction PDFs by headings and subsections.
    """
    # Detect headers like "Line 1.", "General Instructions", "Worksheet..."
    headers = re.split(
        r'(?=^((?:Line \d+[a-z]?\.)|(?:General Instructions)|(?:Specific Instructions)|(?:Worksheet)|(?:Example))\s*)',
        text,
        flags=re.MULTILINE | re.IGNORECASE
    )

    chunks = []
    current_chunk = []

    for part in headers:
        if not part or not part.strip():  # skip None or empty
            continue

        # If it's a header, start a new chunk
        if re.match(r'^(Line \d+[a-z]?\.|General Instructions|Specific Instructions|Worksheet|Example)', part.strip(), flags=re.IGNORECASE):
            if current_chunk:
                chunks.append(" ".join(current_chunk).strip())
            current_chunk = [part.strip()]
        else:
            current_chunk.append(part.strip())

    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())

    # Secondary splitting for oversized chunks
    final_chunks = []
    for c in chunks:
        words = c.split()
        if len(words) > chunk_size:
            for i in range(0, len(words), chunk_size - overlap):
                final_chunks.append(" ".join(words[i:i+chunk_size]))
        else:
            final_chunks.append(c)

    return final_chunks

# ==============================
# 5. Only create embeddings if empty
# ==============================
if collection.count() == 0:
    print("⚡ No embeddings found in ChromaDB. Creating now...")

    # project root = go 2 levels up from this file
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    pdf_path = os.path.join(BASE_DIR, "data", "instructions", "8027.pdf")
    pdf_data = extract_pdf_text(pdf_path)
    print(f"✅ Extracted {len(pdf_data)} pages from PDF")

    all_texts, all_metas, all_ids = [], [], []

    for item in pdf_data:
        # chunks = chunk_text(item["content"])
        chunks = chunk_text(item["content"], chunk_size=400, overlap=50)
        for idx, chunk in enumerate(chunks):
            all_texts.append(chunk)
            all_metas.append({"page": item["page"], "chunk": idx})
            all_ids.append(f"page_{item['page']}_chunk_{idx}")

    embeddings = model.encode(all_texts).tolist()
    print(f"✅ Created {len(all_texts)} new embeddings")

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
# def vector_search(query: str, top_k=5):
#     print(f"\n🔍 Vector search for: {query}")
#     query_vec = model.encode([query])[0].tolist()

#     results = collection.query(
#         query_embeddings=[query_vec],
#         n_results=top_k
#     )

#     context = ""
#     for i, (doc, meta, distance) in enumerate(
#         zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
#     ):
#         # cosine distance → similarity = 1 - distance
#         similarity = 1 - distance
#         page = meta.get("page", "Unknown")
#         snippet = doc[:400]
#         context += (
#             f"Result {i+1}\n"
#             f"📄 Page: {page}, Chunk: {meta.get('chunk', '?')}\n"
#             f"Similarity Score: {similarity:.4f}\n"
#             f"{snippet}...\n---\n"
#         )
#     print("\n=== Retrieved Context ===\n", context)
#     return context
def vector_search(query: str, top_k=3, similarity_threshold=0.6):
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
        similarity = 1 - distance  # cosine distance → similarity score (0–1)

        # ✅ Only include if above threshold
        if similarity >= similarity_threshold:
            page = meta.get("page", "Unknown")
            snippet = doc.strip()[:400]
            context += (
                f"Result {i+1}\n"
                f"📄 Page: {page}, Chunk: {meta.get('chunk', '?')}\n"
                f"Similarity Score: {similarity:.4f}\n"
                f"{snippet}...\n---\n"
            )
        else:
            print(f"⏩ Skipped Result {i+1} (similarity={similarity:.4f} < {similarity_threshold})")

    if not context:
        print("⚠️ No results passed the threshold. Returning empty context.")
    else:
        print("\n=== Filtered Context ===\n", context)

    return context

# ==============================
# 7. Clean Response
# ==============================
# def clean_response(response: str) -> str:
#     response = re.sub(r"^As TaxEase\.AI.*?:", "", response, flags=re.IGNORECASE | re.DOTALL)
#     response = response.replace("*", "").replace("#", "")
#     response = re.sub(r"^\s*[-•]\s*", "", response, flags=re.MULTILINE)
   
#     response = re.sub(r"^\s*\d+\.\s*", "", response, flags=re.MULTILINE)  # Remove "1.", "2.", etc.

#     response = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", response)
#     response = re.sub(r"\[.*?\]", "", response)
#     response = re.sub(r"\((?!Form).*?\)", "", response)
#     response = re.sub(r"\n{2,}", "\n\n", response)
#     response = re.sub(r"[ \t]+", " ", response)
#     return response.strip()
def clean_response(response: str) -> str:
    response = re.sub(r"^As TaxEase\.AI.*?:", "", response, flags=re.IGNORECASE | re.DOTALL)
    response = response.replace("*", "").replace("#", "")
    response = re.sub(r"^\s*[-•]\s*", "", response, flags=re.MULTILINE)
    response = re.sub(r"^\s*\d+\.\s*", "", response, flags=re.MULTILINE)  # Remove "1.", "2.", etc.
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

origins = ["http://localhost:3000", "https://your-ngrok-url.ngrok-free.app","https://tax-front.vercel.app"]

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
