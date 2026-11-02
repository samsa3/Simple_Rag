import ollama
import os, re, json
import numpy as np

DOCS_DIR = "docs"                  # folder with your .txt / .md files
CACHE_DIR = "cache"
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

def load_documents_from_directory(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt") or filename.endswith(".md"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                documents.append(file.read())
    return documents

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    paragraphs = [p.strip() for p in re.split('\n\s*\n', text) if p.strip()]
    chunks, current_chunk = [], ""
    for paragraph in paragraphs:
        while len(paragraph) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            chunks.append(paragraph[:chunk_size])
            paragraph = paragraph[chunk_size - overlap:]
        if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
            current_chunk += ("\n\n" + paragraph) if current_chunk else paragraph
        else:
            chunks.append(current_chunk)
            tail = current_chunk[-overlap:] if overlap else ""
            current_chunk = tail + ("\n\n" + paragraph) if tail else paragraph
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def build_chunks(documents):
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_text(doc))
    return all_chunks



def add_chunk_to_database(chunk,database):
  embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
  database.append((chunk, embedding))

dataset = build_chunks(load_documents_from_directory(DOCS_DIR))


if not any(os.scandir(CACHE_DIR)):  
    VECTOR_DB = []
    for i, chunk in enumerate(dataset):
        add_chunk_to_database(chunk, VECTOR_DB)
        print(f'Added chunk {i+1}/{len(dataset)} to the database')

else:
    Input = input("Embeddings already exist. Do you want to use the existing embeddings? (y/n): ")
    if Input.lower() == 'y':
        VECTOR_DB = np.load(CACHE_DIR, allow_pickle=True).tolist()
        print("Using existing embeddings.")
    elif Input.lower() == 'n':
        VECTOR_DB = []
        print("Creating new embeddings.")
        for i, chunk in enumerate(dataset):
            add_chunk_to_database(chunk, VECTOR_DB)
            print(f'Added chunk {i+1}/{len(dataset)} to the database')


np.save(CACHE_DIR, np.array(VECTOR_DB, dtype=object))