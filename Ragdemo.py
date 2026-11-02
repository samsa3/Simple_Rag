import ollama
import numpy as np

EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'
EMB_PATH = "embeddings.npy"

VECTOR_DB = np.load(EMB_PATH, allow_pickle=True).tolist()

def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    return dot_product / (norm1 * norm2)

def retrieve_similar_chunks(query, top_k=5):
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
    similarities = [(chunk, cosine_similarity(query_embedding, embedding)) for chunk, embedding in VECTOR_DB]
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

input_query = input("I am the cat man. Ask me anything about cats: ")
retrieved_knowledge = retrieve_similar_chunks(input_query)

Instruction_prompt = f""" you are a cat expert. Use the following knowledge to answer questions. \n\n 
knowledge: {'\n'.join([f' - {chunk}' for chunk, _ in retrieved_knowledge])} 
"""

stream = ollama.chat(model=LANGUAGE_MODEL, 
                     messages=[
                        {"role": "user", "content": input_query},
                        {"role": "system", "content": Instruction_prompt}
                        ],
                      stream=True)

print('Chatbot response:')
for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)