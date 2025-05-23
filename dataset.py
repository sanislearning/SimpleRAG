import ollama

dataset = []
with open('cat-facts.txt', 'r', encoding='utf-8') as file:
  dataset = file.readlines()
  print(f'Loaded {len(dataset)} entries')

embedding_model="hf.co/CompendiumLabs/bge-base-en-v1.5-gguf"
language_model="hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF"

#VectorDB is going to hold tuple elements, (chunk, embedding)
#chunk is going to be just a line of text
#embedding is going to be a list of floats that is the semantic vector of a particular chunk
#think dictionary, chunk be the key and embedding be the value

vectorDB=[]

def add_chunk_to_database(chunk):
    embedding=ollama.embed(model=embedding_model,input=chunk)['embeddings'][0]
    #So we use the embed method from ollama, we use chunk as the input
    #We use our preselected embedding model as well
    #Extracts first embedding vector from embeddings list
    vectorDB.append((chunk,embedding))

for i, chunk in enumerate(dataset):
    add_chunk_to_database(chunk)
    print(f'Added chunk {i+1}/{len(dataset)} to the database')