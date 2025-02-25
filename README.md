---

# **📄 PDF Query Chatbot**
A **PDF-based AI chatbot** that allows users to **upload Nepali PDFs**, extract their content, generate **embeddings**, store them in a **FAISS vector database**, and perform **semantic search** using **OpenRouter LLM**.

## 🎯 **Features**
✅ **Upload Nepali PDF** and extract text  
✅ **Supports legacy Nepali fonts** (Preeti, Kantipur, Sagarmatha)  
✅ **Automatically converts text to Unicode**  
✅ **Splits text into chunks for better retrieval**  
✅ **Embeds chunks using Sentence Transformers**  
✅ **Stores embeddings in FAISS for fast search**  
✅ **Supports English & Nepali queries**  
✅ **Uses OpenRouter LLM (DeepSeek-70B) for AI responses**  
✅ **Translates queries and responses via Google Translate**  

---

## 🚀 **Installation & Setup**
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/manjil-budhathoki/pdf-query-chatbot
cd pdf-query-chatbot
```

### **2️⃣ Create a Virtual Environment (Optional but Recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### **3️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4️⃣ Set Up API Keys**
1. Create a `.env` file in the root directory.
2. Add your **OpenRouter API Key**:
   ```
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   ```
---

## 🎯 **Usage**
### **Run the Chatbot**
```bash
streamlit run Main.py
```

### **How It Works**
1️⃣ **Upload a Nepali PDF**  
2️⃣ The app **extracts and converts** text to Unicode  
3️⃣ The text is **split into chunks** for efficient retrieval  
4️⃣ **SentenceTransformer** generates **embeddings**  
5️⃣ FAISS stores embeddings in a `.pkl` file for **fast access**  
6️⃣ When the user **asks a question**, it:  
   - Converts the query to **Nepali**  
   - Computes **query embeddings**  
   - Performs **semantic search** in FAISS  
   - Retrieves the **most relevant text**  
7️⃣ The context is sent to **OpenRouter API (DeepSeek-70B)**  
8️⃣ AI **generates a concise answer**  
9️⃣ If requested, the response is **translated to Nepali**  

---

## 📜 **Project Structure**
```
📦 pdf-query-chatbot/
├── 📜 .env                 # Stores API keys (not tracked in GitHub)
├── 📜 Main.py              # Streamlit chatbot interface
├── 📜 map.json             # Legacy Nepali font mapping
├── 📜 README.md            # Documentation
├── 📜 requirements.txt      # List of dependencies
```

---

## ⚙ **Configuration**
### **FAISS Embedding Storage**
- Stores FAISS embeddings in a `.pkl` file  
- If embeddings exist, **loads from disk** instead of recomputing  

### **OpenRouter API**
- Uses **DeepSeek-70B** for AI responses  
- You must **set up your OpenRouter API Key** in `.env`  

---

## 🛠 **Troubleshooting**
#### **1. OpenRouter API Key Not Found?**
- Ensure the `.env` file is correctly placed and formatted.
- Run:
  ```bash
  echo $OPENROUTER_API_KEY  # On Windows: echo %OPENROUTER_API_KEY%
  ```
- If missing, manually add:
  ```
  OPENROUTER_API_KEY=your_api_key_here
  ```

#### **2. ModuleNotFoundError?**
- Run:
  ```bash
  pip install -r requirements.txt
  ```

#### **3. PDF Text Not Extracted?**
- The PDF **may contain images** instead of text. Try **another PDF**.

#### **4. AI Response is Too Long?**
- Modify the system prompt in `query_openrouter()`:
  ```python
  prompt = f"""
  Context:
  {context}

  User Question:
  {query}

  Instructions:
  - Provide a concise and direct response.
  - Use contract-style writing (brief, bullet points if applicable).
  - Avoid unnecessary explanations.
  - If additional information is needed, state "Not enough data in context."
  """
  ```

---

## 🌟 **Contributing**
Feel free to **fork the repo**, submit **pull requests**, or report **issues**!  
Let's improve this **Nepali PDF chatbot together**. 🚀  

---

## 📜 **License**
This project is **open-source** under the **MIT License**.

---

### 🎯 **Now You're Ready to Run the Chatbot!**
```bash
streamlit run Main.py
```
🚀 **Enjoy querying your Nepali PDFs!** 😊  

---
