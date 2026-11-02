import ollama
import numpy as np
import re

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

def llm_score(query, chunk):
    prompt = f"""rank the following knowledge based on its relevance to the question. {query} \n\n knowledge: {chunk} \n\n
    Return a score between 0 and 10, where 10 is highly relevant and 0 is not relevant at all. reply with only a number \n\n"""
    response = ollama.chat(model=LANGUAGE_MODEL, 
                         messages=[
                            {"role": "user", "content": prompt},
                            ],)
    match = re.search(r"\d+(\.\d+)?", response["message"]["content"])
    
    return min(float(match.group()), 10.0) if match else 0.0

def rerank_chunks(query, chunks, top_k=5):
    scored_chunks = [(chunk, llm_score(query, chunk)) for chunk, _ in chunks]
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    return scored_chunks[:top_k]


input_query = input("I am the cat man. Ask me anything about cats: ")
retrieved_knowledge = rerank_chunks(input_query, retrieve_similar_chunks(input_query, top_k=10), top_k=5)
print(rerank_chunks(input_query, retrieve_similar_chunks(input_query, top_k=10), top_k=5))

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