from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchian.text_splitter import RecursiveCharacterTextSplitter 
import faiss
import numpy as np

from transformers import pipeline

#reading a pdf and extracting the text
reader = PdfReader('C:/Users/Amosh/Downloads/qcnotes.pdf')
page = reader.pages[0]
text = page.extract_text()
print(len(text))


#breaking the extracted text into smaller portions of 1000 characters and appending into a list.
splitter = RecursiveCharacterTextSplitter(
    chunksize = 500,
    overlap = 100
)
chunks = splitter.split_text(text)

#embedding the chunks in vector model

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)
print(embeddings.shape)


#stored the embeddings into vectors
embedding_array = np.array(embeddings).astype('float32')
dimension = embedding_array.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embedding_array)

#user's query converting to vector
query = input("Ask anything")
query_embeddings = model.encode([query])

#performing semantic search
k = 3 #top 3 searches
distance, indices = index.search(query_embeddings, k)


#creating context
context =""

for idx in indices[0]:
    context += chunks[idx] +"\n" 

prompt = f""" 
Answer the question based on the context below.
context:
{context}

Question: 
{query}

Answer:
"""


#creating LLM
generator = pipeline(
    "text2text-generation",
    model = "google/flan-t5-base"
)

#Generating final response
response = generator(
    prompt,
    max_length = 200
)
print(response[0]['generated_text'])