import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# ==============================
# STEP 1: Extract Text from PDF
# ==============================
def extract_pdf_text(file_path):
    reader = PdfReader(file_path)
    data = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            data.append({"page": i+1, "content": text})
    return data

pdf_path = r"D:\tax\TaxEase.AI-Vertex-AI-Agent\data\instructions\1040-SE.pdf"
pdf_data = extract_pdf_text(pdf_path)

print(f"✅ Extracted {len(pdf_data)} pages from PDF")

# ==============================
# STEP 2: Create Embeddings
# ==============================
model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [item["content"] for item in pdf_data]
embeddings = model.encode(texts)

print("✅ Created embeddings")

# ==============================
# STEP 3: Store in ChromaDB
# ==============================
client = chromadb.Client(Settings(persist_directory="./chroma_db"))
collection = client.get_or_create_collection(name="pdf_embeddings")

collection.add(
    documents=texts,
    embeddings=embeddings.tolist(),
    metadatas=pdf_data,
    ids=[f"page_{i}" for i in range(len(texts))]
)

print("✅ Stored PDF embeddings in ChromaDB")

# ==============================
# STEP 4: Query Example
# ==============================
query = "What is the self-employment tax rate?"
query_embedding = model.encode([query])[0]

results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=3
)

print("🔍 Search Results:")
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"\n📄 Page {meta['page']}: {doc[:200]}...")
