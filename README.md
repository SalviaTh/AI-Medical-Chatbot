# AI Medical Chatbot

An intelligent, Retrieval-Augmented Generation (RAG) powered medical chatbot. This application utilizes a medical encyclopedia to provide accurate, context-aware answers to user health queries.

## Features
- **Retrieval-Augmented Generation (RAG):** Grounds the AI's responses in actual medical literature to prevent hallucinations.
- **Google Gemini 2.0 Flash:** High-performance, fast LLM for conversational responses.
- **Pinecone Vector Database:** Efficient storing and semantic similarity searching of medical text chunks.
- **Modern UI:** A beautiful, responsive, dark-mode styling with glassmorphism effects and asynchronous typing indicators.

## Tech Stack
- **Backend Framework:** Python Flask
- **LLM Orchestration:** LangChain (using modern LangChain Expression Language - LCEL)
- **Generative AI:** Google Gemini (`gemini-2.0-flash`)
- **Embeddings:** HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Database:** Pinecone
- **Frontend:** HTML, CSS (Vanilla), minimal JavaScript (Fetch API)

## 📁 Project Structure 
```
.
├── Data/                 # Put your medical PDF files here
├── src/                  
│   ├── helper.py         # Functions for loading PDFs, splitting text, and embedding
│   └── prompt.py         # System prompt constraining the AI to medical QA
├── static/
│   └── style.css         # Modern, premium UI styling
├── templates/
│   └── index.html        # Front-end chatbot interface
├── app.py                # Main Flask application with LCEL RAG chain
├── store_index.py        # Script to chunk data and push to Pinecone
├── requirements.txt      # Python dependencies
└── Dockerfile            # Production Docker setup
```

##  How to Setup and Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/SalviaTh/AI-Medical-Chatbot.git
cd AI-Medical-Chatbot
```

### 2. Set up your Virtual Environment
```bash
python -m venv myenv
# For Windows:
myenv\Scripts\activate
# For Mac/Linux:
source myenv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup Environment Variables
Create a `.env` file in the root directory and add your API keys:
```env
PINECONE_API_KEY="your-pinecone-api-key-here"
GEMINI_API_KEY="your-google-gemini-api-key-here"
```

### 5. Create the Vector Database Index (One-Time Setup)
Ensure you have your medical reference PDF inside the `Data/` folder. Then, generate your embeddings and push them to your Pinecone index by running:
```bash
python store_index.py
```

### 6. Start the Chatbot
```bash
python app.py
```
Open **http://localhost:8080** in your browser to interact with the assistant!

## License
This project is for educational and portfolio purposes. 

> **Disclaimer:** This AI is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
