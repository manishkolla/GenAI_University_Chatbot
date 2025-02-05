"""
This script integrates various machine learning and natural language processing (NLP) techniques to build an intelligent 
chatbot for Georgia State University (GSU). It utilizes Google's Generative AI (google.generativeai), 
the SentenceTransformer model for text embeddings, and FAISS for efficient similarity searches. 
The script begins by ensuring necessary NLTK resources (such as stopwords, tokenizers, and lemmatizers) 
are available. It defines preprocessing steps like text normalization, tokenization, stopword 
removal, and lemmatization, followed by summarization using TF-IDF ranking. The chatbot, 
configured with predefined prompts, leverages the Gemini API for responses and integrates 
Bing Search for external references. A key function, corrective_rag, refines retrieved text 
relevance by classifying it as Correct, Ambiguous, or Wrong, generating JSON-formatted output. 
Additionally, the script includes a convert_to_chunks function to tokenize large text inputs efficiently. 
This setup enables the chatbot to provide accurate university-related responses by retrieving, 
summarizing, and refining information dynamically.
"""

import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd

import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from heapq import nlargest
from config import GEMINI_API, BING_API
import nltk
import nltk

# Define a function to check and download each resource if necessary
def download_nltk_resource(resource_name):
    try:
        nltk.data.find(f"tokenizers/{resource_name}" if resource_name == "punkt" else f"corpora/{resource_name}")
        print(f"{resource_name} is already installed.")
    except LookupError:
        print(f"{resource_name} not found. Downloading...")
        nltk.download(resource_name)

# List of resources to check and download if missing
resources = ['stopwords', 'punkt', 'wordnet', 'punkt_tab']

for resource in resources:
    download_nltk_resource(resource)

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('punkt_tab')
# Load the Sentence Transformer model
model = SentenceTransformer('bert-base-nli-mean-tokens')

# with open(r"C:/Users/mkolla1/OneDrive - Georgia State University/Desktop/CareerSwipe/Google_AI_API.txt", "r") as f:
#     GOOGLE_API_KEY = f.read()
genai.configure(api_key=GEMINI_API)
prompt= """ You are Georgia State University's dedicated website chatbot. Your primary function is to provide accurate and helpful information about Georgia State University. You will draw on your knowledge base and the provided reference material {reference_text} to answer user queries.

Please strictly adhere to the following guidelines:

GSU Focus: Limit your responses to topics directly related to Georgia State University, for any other questions please respectfully tell the user to ask questions related to the universiy and you cannot help beyond that.
Informative and Concise: Provide clear, concise, and relevant information.
Helpful and Respectful: Maintain a positive and helpful tone, treating all users with respect.
External Reference: If a query requires information beyond your current knowledge, provide relevant links to official GSU websites or other credible sources.
Example:

User: What are the tuition fees at GSU?
You: Tuition fees at Georgia State University vary depending on your program and residency status. Please visit the official tuition and fees page for the most accurate information: [link to GSU tuition and fees page]"""
prompt1="""You are Georgia State University's Website Chatbot, you will be assisting with any type of questions the user might have about 
Georgia State University, you will be assisting with your own knowledge and some reference materials provided to you, here is the reference material {reference_text}
Please provide with any additional links and reference URL's for the users. The user will be asking questions and please be kind and respectful to the user. Good Luck!"""
import os
import glob
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF for PDF processing

# Initialize the SentenceTransformer model
model = SentenceTransformer('bert-base-nli-mean-tokens')

import os 
from pprint import pprint
import requests
def bing_search(query): 
    subscription_key = BING_API
    endpoint = "https://api.bing.microsoft.com/v7.0/search"


    # Construct a request
    mkt = 'en-US'
    params = {"q": query, "textDecorations": True, "textFormat": "HTML"}
    #params = { 'q': query, 'mkt': mkt }
    headers = { 'Ocp-Apim-Subscription-Key': subscription_key }

    # Call the API
    try:
        response = requests.get(endpoint, headers=headers, params=params)
        response.raise_for_status()

        # print(" Headers:")
        # print(response.headers)

        #print("JSON Response:")
        #pprint(response.json())
        search_results= response.json()
    
        rows = "\n".join(["""<tr>
                       <td><a href=\"{0}\">{1}</a></td>
                       <td>{2}</td>
                     </tr>""".format(v["url"], v["name"], v["snippet"])
                  for v in search_results["webPages"]["value"]])
        return rows
    except Exception as ex:
        print("error")
        print(ex)



# Download necessary NLTK resources (uncomment if running for the first time)
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

def preprocess_text(text):
    # Step 1: Convert to lowercase
    text = text.lower()

    # Step 2: Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Step 3: Tokenization
    tokens = word_tokenize(text)

    # Step 4: Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Step 5: Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Step 6: Summarization (optional, reduces text further)
    summarized_text = summarize_text(' '.join(tokens))

    return summarized_text

def summarize_text(text, top_n=5):
    """
    Summarizes text by selecting the most important sentences based on TF-IDF
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform([text])
    
    # Rank sentences based on TF-IDF
    sentences = text.split('.')
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        score = 0
        for word in sentence.split():
            if word in vectorizer.get_feature_names_out():
                score += X[0, vectorizer.get_feature_names_out().tolist().index(word)]
        sentence_scores[i] = score

    # Select the top_n most important sentences
    top_sentences = nlargest(top_n, sentence_scores, key=sentence_scores.get)
    top_sentences = [sentences[i] for i in top_sentences]

    return '. '.join(top_sentences)




import json
import json
import re
from transformers import AutoTokenizer
query2 = """Can you provide me with the director and associate chair of GSU CS Department? and also can you provide me with some helpful link to get a research oppurtunity at GSU Computer Science department"""
query = """Can you provide me with 5 GSU Computer Science Faculty who are working as professors and their emails please. Also who is Dr.Parag Tamhankar from Directory"""

def corrective_rag(query, reference_text):
    prompt = """
    You are a highly accurate and knowledgeable retrieval evaluator. Your task is to assess the relevance of provided reference text to a given query. For each piece of reference text, classify it as one of the following:

Correct: The text directly answers the query, providing accurate and relevant information.
Ambiguous:The text is partially relevant but requires additional context or interpretation to fully answer the query.
Wrong: The text is irrelevant to the query or provides incorrect information.
Generate ouput in a JSON file, here is a sample format: {format}
Even if the reference piece is Ambiguous or Wrong or Correct, please do generate it in the output. 
You will be provided with a query and a list of reference texts. Here are the documents {reference}
    """  # Paste the full prompt here

    corrective_model = genai.GenerativeModel(
        'models/gemini-1.5-flash',
        system_instruction=prompt.format(reference=reference_text, format= [{'file_name': 'file name', 'relevance': 'Correct'}, {'file_name': 'file name', 'relevance': 'Wrong'}, {'file_name': 'file name', 'relevance': 'Ambiguous'}]), generation_config={"response_mime_type": "application/json"}
    )

    response = corrective_model.generate_content(query)
    # Parse the response to extract classifications
    try:
        classifications= json.loads(response.text)
    except Exception as e:
        print(e)
        classifications= response.text
        print(classifications)
    # print(classifications)
    reference_dict = {item['file_name']: item['text'] for item in reference_text}

    # Iterate through classification and add text if file_name matches
    for item in classifications:
        file_name = item['file_name']
        if file_name in reference_dict:
            item['text'] = reference_dict[file_name]
    
    # classification now has the 'text' added where file names matched
    # Filter and refine correct documents
    final_knowledge_chunk = process_chunks(classifications, query)
    
    # Save or return the final knowledge chunk
    with open("refined_knowledge_chunk.json", "w") as file:
        json.dump({"refined_knowledge_chunk": final_knowledge_chunk}, file, indent=2)
    
    return final_knowledge_chunk
    #return classifications

def convert_to_chunks(text, max_tokens=5000):
    # Initialize tokenizer (adjust model name if necessary)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Split text by sentences or paragraphs
    sentences = re.split(r'(?<=[.!?]) +', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # Add sentence to the current chunk and check the token length
        current_chunk += " " + sentence
        tokenized_length = len(tokenizer.encode(current_chunk))
        
        if tokenized_length > max_tokens:
            # Append the chunk if it exceeds max_tokens, then reset
            chunks.append(current_chunk.strip())
            current_chunk = sentence  # Start a new chunk with current sentence

    # Append any remaining text as a final chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def process_chunks(classifications, query):
    knowledge_chunks = []
    status=True
    for item in classifications:
        # Check if the relevance is "Correct"
        if item['relevance'] == 'Correct':
            text = item.get('text', '')  # Assuming 'text' contains the document content
            file_name = item.get('file_name', 'unknown')
            
            # Convert document text to smaller knowledge chunks
            chunks = convert_to_chunks(text)
            #chunks= preprocess_text([text])
            # Append each chunk as a knowledge piece
            for chunk in chunks:
                knowledge_chunks.append({
                    "file_name": file_name,
                    "chunk_text": chunk
                })
        if item['relevance']== 'Wrong' or item['relevance']== 'Ambiguous' :
            while status:
                print("BING SEARCH")
                text = item.get('text', '')  # Assuming 'text' contains the document content
                file_name = item.get('file_name', 'unknown')

                chunks= bing_search(query)
                chunks= preprocess_text(chunks)
                knowledge_chunks.append({
                        "file_name": 'Bing Search Results',
                        "chunk_text": chunks
                    })
                status=False
    return knowledge_chunks

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
        D, I = index.search(query_embedding, k=5)  # Retrieve top 3 matches
        
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


history=[]
def answer(query):
    global history
    top_results = search_query(query, indices, doc_texts, model)
    print(len(top_results))
    classification= corrective_rag(query,top_results )
    print(len(classification))
    # Top result text for sending to Gemini bot
    if top_results:
        top_result_text = top_results[0]['text']
    print("Got the Reference Text")
    #print(classification)
    ai_model = genai.GenerativeModel('models/gemini-1.5-flash', 
                                    system_instruction=prompt.format(reference_text=classification))
    chat=ai_model.start_chat(history=history)
    history.append({'role':'user', 'parts':query})
    response = chat.send_message(query)
    history.append({'role':'model', 'parts':response.text})
    print(response.text)
    return response.text


query2 = """Can you provide me with the director and associate chair of GSU CS Department? and also can you provide me with some helpful link to get a research oppurtunity at GSU Computer Science department"""
query = """Can you provide me with 5 GSU Computer Science Faculty who are working as professors and their emails please. Also who is Dr.Parag Tamhankar from Directory"""
query3="""organic and inorganic chemistry research professors  at Georgia State University """
query4=""" Can you give me a list of 5 GSU Chemistry lecturers"""
# answer(query2)