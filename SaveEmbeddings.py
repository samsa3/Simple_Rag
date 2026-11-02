import ollama
import os
import numpy as np

dataset = []
with open ("cat-facts.txt", "r") as file:
    dataset = file.readlines()
print (f"Loaded {len(dataset)} entries")

EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
EMB_PATH = "embeddings.npy"

def add_chunk_to_database(chunk,database):
  embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
  database.append((chunk, embedding))

if os.path.exists(EMB_PATH):
    Input = input("Embeddings already exist. Do you want to use the existing embeddings? (y/n): ")
    if Input.lower() == 'y':
        VECTOR_DB = np.load(EMB_PATH, allow_pickle=True).tolist()
        print("Using existing embeddings.")
    elif Input.lower() == 'n':
        VECTOR_DB = []
        print("Creating new embeddings.")
        for i, chunk in enumerate(dataset):
            add_chunk_to_database(chunk, VECTOR_DB)
            print(f'Added chunk {i+1}/{len(dataset)} to the database')
else:
    VECTOR_DB = []
    for i, chunk in enumerate(dataset):
        add_chunk_to_database(chunk, VECTOR_DB)
        print(f'Added chunk {i+1}/{len(dataset)} to the database')

np.save(EMB_PATH, np.array(VECTOR_DB, dtype=object))