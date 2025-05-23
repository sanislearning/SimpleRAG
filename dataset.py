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

#Next we will deal with the implemenatation of the retrieval function

def cosine_similarity(a,b):
    dot_product=sum([x*y for x,y in zip(a,b)])
    norm_a=sum([x**2 for x in a])**0.5
    norm_b=sum([x**2 for x in b])**0.5
    return dot_product/(norm_a*norm_b)
    #https://www.youtube.com/watch?v=e9U0QAFbfLI, check this out if you wanna figure out implementation

def retrieve(query,top_n=3):
    query_embedding=ollama.embed(model=embedding_model,input=query)['embeddings'][0]
    similarities=[] #Creates a list that will store tupes of (chunk, similarity_score)
    for chunk,embedding in vectorDB:
        similarity=cosine_similarity(query_embedding,embedding)
        similarities.append((chunk,similarity))
        #sorting by similarity in desecending order, higher the value of similarity the more relevant the chunk
        similarities.sort(key=lambda x:x[1], reverse=True)
        return similarities[:top_n]