# A New Generative Approach to Optimize the Network and Server Load of Websites (University Domanin Chatbot)
### Project submitted to Dr. Yanqing Zhang for Artificial Intelligence (CS4810)
### Authors: Manish Kolla & Ritesh Dumpala

# **Application Manual**

This manual provides step-by-step instructions to set up and run the University Chatbot application locally.

---
Here's the updated version of the **Prerequisites** section with steps to check for Python and Visual Studio Code installations, and install them if necessary:

## **Prerequisites**

Before running the application, ensure you have the following:

1. **Python Installation**  
   The application requires Python. To check if Python is installed, run the following command in your terminal or command prompt:
   ```bash
   python --version
   ```
   If Python is not installed, follow these steps to install it:
   - Download Python from the official [Python website](https://www.python.org/downloads/).
   - Run the installer and make sure to check the box that says **Add Python to PATH** during the installation process.
   - After installation, confirm Python is installed by running:
   ```bash
   python --version
   ```

2. **Visual Studio Code (VS Code)**  
   Visual Studio Code is recommended as the code editor for this application. To check if VS Code is installed, run the following command in your terminal:
   ```bash
   code --version
   ```
   If VS Code is not installed, follow these steps to install it:
   - Download the installer for your operating system from the official [VS Code website](https://code.visualstudio.com/Download).
   - Run the installer and follow the prompts to complete the installation.
   - Once installed, you can open VS Code by typing `code` in the terminal.

3. **Gemini API Key**  
   The Gemini API key is required for interacting with Generative AI. Follow these steps to obtain one:
   - Login to the [Gemini AI Studio](https://aistudio.google.com/).
   - Sign in with your account or create a new one.
   - Click on **Get API Key** section and follow the prompts.
   - Generate a new API key and copy it.  
     *Note: Save the key securely as it will be used in the application.*

4. **Bing Search API Key**  
   The Bing Search API key is required for integrating web search functionality. Follow these steps to obtain one:
   - Visit the [Microsoft Azure Portal](https://portal.azure.com/).
   - Sign in with your Microsoft account or create a new one.
   - Go to **Create a Resource** and search for "Bing Search v7".
   - Set up a new resource for Bing Search and navigate to the **Keys and Endpoint** section.
   - Copy one of the provided API keys.  
     *Note: Save the key securely as it will be used in the application.*

5. **Create the `config.py` File**  
   To securely store your API keys, create a `config.py` file in the project directory and add the following code:
   ```python
   # config.py
   GEMINI_API = "YOUR_API_KEY_HERE"
   BING_API = "YOUR_API_KEY_HERE"
   ```

   - Replace `"YOUR_API_KEY_HERE"` with your actual **Gemini API Key** and **Bing API Key**.
   - Save this file in the same directory as your main application script (`app.py`).
---
## **Steps to Run the Application**

### 1. **Clone the Repository**
Use the following command to clone the repository from your Git hosting platform (e.g., GitHub):  
```bash
git clone https://github.com/manishkolla/GenAI_University_Chatbot
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


