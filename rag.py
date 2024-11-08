

import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
# Load the Sentence Transformer model
model = SentenceTransformer('bert-base-nli-mean-tokens')

with open(r"C:/Users/mkolla1/OneDrive - Georgia State University/Desktop/CareerSwipe/fakekey.txt", "r") as f:
    GOOGLE_API_KEY = f.read()
genai.configure(api_key=GOOGLE_API_KEY)


import os
import glob
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF for PDF processing

# Initialize the SentenceTransformer model
model = SentenceTransformer('bert-base-nli-mean-tokens')

# Dictionary to store FAISS indices and corresponding document texts
indices = {}
doc_texts = {}

# Helper function to create FAISS index and store text and embeddings
def create_and_store_index(text, file_name):
    embeddings = model.encode([text])
    embeddings = np.array(embeddings).astype('float32')
    
    # Initialize FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Store the index and document text in dictionaries
    indices[file_name] = index
    doc_texts[file_name] = text
    print(f"Index created for '{file_name}' with {index.ntotal} vectors.")

# Function to perform a search across all indices and return top result text
def search_query(query, indices, doc_texts, model):
    # Encode the query into an embedding
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype('float32')
    
    top_result_texts = []  # List to store top results with filenames and distances
    
    # Search each FAISS index
    for file_name, index in indices.items():
        # Search for the nearest neighbors in the current index
        D, I = index.search(query_embedding, k=5)  # Retrieve top 5 matches
        
        # Collect results for this file
        for i in range(3):
            top_result_texts.append({
                'file_name': file_name,
                'text': doc_texts[file_name],  # Document text for each index
                'distance': D[0][i]  # Distance to the match
            })
    
    # Sort results by the closest distance across all files
    sorted_results = sorted(top_result_texts, key=lambda x: x['distance'])
    
    # Display the top 3 matches
    for i, result in enumerate(sorted_results[:5]):
        print(f"\nTop {i+1} match in '{result['file_name']}':")
        print("Distance:", result['distance'])
    
    return sorted_results[:5]  # Return only the top 3 results

# Folder path where PDF and TXT files are stored
data_folder = "data"

# Retrieve all PDF and TXT files from the folder
pdf_files = glob.glob(os.path.join(data_folder, "*.pdf"))
txt_files = glob.glob(os.path.join(data_folder, "*.txt"))

# Process PDF files and store their indices and texts
for pdf_file in pdf_files:
    with fitz.open(pdf_file) as doc:
        text = "".join([page.get_text() for page in doc])
    create_and_store_index(text, pdf_file)

# Process TXT files and store their indices and texts
for txt_file in txt_files:
    with open(txt_file, 'r', encoding='utf-8') as file:
        text = file.read()
    create_and_store_index(text, txt_file)





prompt="""You are Georgia State University's Website Chatbot, you will be assisting with any type of questions the user might have about 
Georgia State University, you will be assisting with your own knowledge and some reference materials provided to you, here is the reference material {reference_text}
Please provide with any additional links and reference URL's for the users. The user will be asking questions and please be kind and respectful to the user. Good Luck!"""

def answer(query):
    top_results = search_query(query, indices, doc_texts, model)
    if top_results:
            top_result_text = top_results[0]['text']
        
    ai_model = genai.GenerativeModel('models/gemini-1.5-flash', 
                                        system_instruction=prompt.format(reference_text=top_result_text))

    response = ai_model.generate_content(query)
    print(response.text)


query= "Can you give me a list of 5 GSU Chemistry lecturers"
answer(query)