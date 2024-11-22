# A-New-generative-Approach-to-Optimize-the-Network-and-Server-Load-of-Websites Manual


# **Application Manual**

This manual provides step-by-step instructions to set up and run the University Chatbot application locally.

---

## **Steps to Run the Application**

### 1. **Clone the Repository**
Use the following command to clone the repository from your Git hosting platform (e.g., GitHub):  
```bash
git clone <repository-url>
```

### 2. **Navigate to the Project Directory**
Change your working directory to the cloned repository folder:  
```bash
cd <project-directory>
```

### 3. **Check Python and pip Installation**
Make sure Python and pip are installed on your system.  

#### To check if Python is installed:  
```bash
python --version
```
or  
```bash
python3 --version
```

#### To check if pip is installed:  
```bash
pip --version
```

If either is missing, download and install the latest version of Python from the [official Python website](https://www.python.org/).

---

### 4. **Create and Activate a Virtual Environment**
It is recommended to use a virtual environment to isolate dependencies.  

#### For **Windows**:
1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
2. Activate the virtual environment:
   ```bash
   venv\Scripts\activate
   ```

#### For **Mac/Linux**:
1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   ```
2. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

---

### 5. **Install Required Dependencies**
Install all necessary Python libraries listed in the `requirements.txt` file:  
```bash
pip install -r requirements.txt
```

---

### 6. **Run the Flask Application**
Run the Flask application using the following command:  
```bash
python app.py
```

---

### 7. **Access the Application Locally**
Once the application starts, it will display a local URL (usually `http://127.0.0.1:5000/`). Open this URL in your web browser to access the application.

---

### **8. Test the Application**
You can test the application by interacting with it through the web interface or API (depending on the implementation).

- Open the URL shown in the terminal (e.g., `http://127.0.0.1:5000/`) in your browser.  
- Ask questions related to **Computer Science (CS)**, **Data Science departments**, or the **directory**. For example:
  - "What is the role of the CS department?"
  - "Can you provide information about the Data Science program?"
  - "Who is the head of the CS department?"
  - "Can you find a contact in the directory?"
- The application should respond with the relevant information or provide helpful answers based on its functionality.

---

### **Optional: Deactivate the Virtual Environment**
Once done, deactivate the virtual environment using:  
```bash
deactivate
```


