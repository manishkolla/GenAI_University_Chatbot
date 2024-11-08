
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
#from sentence_transformers import SentenceTransformerimport
import faiss
import os
import os
import glob
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF for PDF processing


# Load the Sentence Transformer model
model = SentenceTransformer('bert-base-nli-mean-tokens')

# Dictionary to store FAISS indices and corresponding document texts
indices = {}
doc_texts = {}
embeddings_list = []  # List to store embeddings and filenames for CSV export

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
    
    # Add embeddings and file name to the list for CSV export
    embeddings_list.append({'file_name': file_name, 'embedding': embeddings.flatten().astype('float32'), 'text': text})
    
    print(f"Index created for '{file_name}' with {index.ntotal} vectors.")

# Function to save embeddings to CSV
def save_embeddings_to_csv(embeddings_list, csv_file="embeddings.csv"):
    # Check if embeddings_list is structured correctly
    if not embeddings_list:
        print("No embeddings to save.")
        return

    # Convert the embeddings list into a DataFrame
    embeddings_df = pd.DataFrame(embeddings_list)
    
    # Save the DataFrame to a CSV file
    embeddings_df.to_csv(csv_file, index=False)
    print(f"Embeddings saved to '{csv_file}'.")

# Folder path where PDF and TXT files are stored
#data_folder = r"data"
data_folder = os.path.abspath("data")  # Converts relative path to absolute path

# Check if the folder exists
if not os.path.exists(data_folder):
    print(f"The folder {data_folder} does not exist.")
else:
    print(f"The folder {data_folder} exists.")
# Retrieve all PDF and TXT files from the folder
pdf_files = glob.glob(os.path.join(data_folder, "*.pdf"))
txt_files = glob.glob(os.path.join(data_folder, "*.txt"))
print(pdf_files)
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

# Save embeddings to CSV after processing all files
save_embeddings_to_csv(embeddings_list)