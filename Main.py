import streamlit as st
import pickle
import fitz  # PyMuPDF for PDF text extraction
import npttf2utf  # For Nepali legacy font conversion
from langdetect import detect
import os
import requests  # For OpenRouter API
import dotenv  # Load API key from .env
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer  # Multilingual embeddings
from langchain.vectorstores import FAISS
from googletrans import Translator  




# OpenRouter API Configuration
dotenv.load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = "deepseek/deepseek-r1-distill-llama-70b:free"

translator = Translator()


class SentenceTransformerEmbeddings:
    """Wrapper class to make SentenceTransformer compatible with FAISS."""
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        """FAISS expects this method for embedding documents."""
        return self.model.encode(texts, convert_to_tensor=False)

    def embed_query(self, text):
        """FAISS also expects this method for query embeddings."""
        return self.model.encode([text], convert_to_tensor=False)[0]

# Load multilingual embedding model in a FAISS-compatible format
embedding_model = SentenceTransformerEmbeddings()

def extract_text_from_pdf(uploaded_pdf):
    """Extracts text from a PDF file using PyMuPDF (fitz)."""
    text = ""
    doc = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")

    for page in doc:
        page_text = page.get_text("text") # change to "block" if needed
        text += page_text + "\n"

    if not text.strip():
        st.error("⚠️ No extractable text found! Your PDF might be an image-based or scanned document.")
    
    return text.strip()

def process_text(text):
    """Detects language and converts Nepali fonts if necessary."""
    try:
        detected_lang = detect(text)  # 'ne' for Nepali, 'en' for English
        if detected_lang == "en":
            return text  # No conversion needed
        
    except:
        st.warning("⚠️ Language detection failed. Assuming Nepali.")

    # Convert Nepali legacy fonts
    font_types = ["Preeti", "Sagarmatha", "Kantipur"]
    mapper = npttf2utf.FontMapper("map.json")

    for font in font_types:
        try:
            unicode_text = mapper.map_to_unicode(text, from_font=font)
            if "?" not in unicode_text and unicode_text.strip():  # Ensure valid conversion
                return unicode_text
        except Exception:
            continue  # Try next font

    return text  # Return original if conversion fails
  
def translate_text(text, target_lang="en"):
    """Translates text using Google Translate."""
    if text.strip():
        return translator.translate(text, dest=target_lang).text
    return text

def query_openrouter(prompt):
    """Sends a query to OpenRouter LLM API and returns the response."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": "You are an assistant that helps answer questions based on provided context."},
            {"role": "user", "content": prompt}
        ]
    }
    
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
    
    if response.status_code != 200:
        st.error(f"⚠️ OpenRouter API Error: {response.status_code} - {response.text}")
        return "Sorry, I couldn't process your request."
    
    return response.json()["choices"][0]["message"]["content"]
      
      
# Sidebar UI
with st.sidebar:
    st.title('📄 PDF Query Chatbot')
    st.markdown('''
    ## About
    This app reads PDFs and ensures correct text rendering:
    - **Streamlit** for UI
    - **PyMuPDF (fitz)** for better text extraction
    - **npttf2utf** for Nepali font conversion
    - **OpenRouter API** for LLM-powered answers
    - **Google Translate API** for translations
    ''')
    add_vertical_space(3)
    st.write('💡 Made with ❤️ by [Manjil Budhathoki]')

# Main UI
def main():
    st.header("🤖 Chat with your Nepali PDF!")

    # Upload PDF file
    pdf = st.file_uploader("📤 Upload your PDF", type="pdf")
    
    # language selection between Nepali and english:
    lang_choice = st.radio("Select Response Language:", ("English", "Nepali"))

    if pdf is not None:
        extracted_text = extract_text_from_pdf(pdf)
        if extracted_text:  # Process only if text is extracted
            processed_text = process_text(extracted_text)

            # st.subheader("📜 Extracted & Processed Text:")
            # st.write(processed_text)
        else:
            st.error("❌ No text found. Try another PDF or check if it's a scanned document.")
            return

         # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
          chunk_size=1000, # which represent token in our case 1000 token each
          chunk_overlap=200, # overlaping token between the consective chunks
          length_function=len
        )
        
        chunks = text_splitter.split_text(text=processed_text)
        
        # st.write(chunks)
        
        # Generate embeddings ONCE and store in FAISS
        # st.subheader("🔢 Generating Embeddings...")
        
        store_name = pdf.name[:4] 

        # Check if FAISS store exists
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                vector_store = pickle.load(f)
            
            # st.success("✅ Embeddings loaded from disk!")
        
        else:
          
            # Store embeddings in FAISS using the wrapped SentenceTransformer
            # vector_store = FAISS.from_texts(chunks, embedding=embedding_model)
            vector_store = FAISS.from_texts(chunks, embedding=embedding_model.embed_documents)

            # Save FAISS store to disk
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(vector_store, f)

            # st.success(f"✅ {len(chunks)} chunks stored in FAISS as `{store_name}.pkl`!")
        
        # Accept User Question/Query
        query = st.text_input("Ask question about your PDF File:")
        # st.write(query)
        
        if query:
            # 🔹 Translate user query to Nepali for better retrieval
            nepali_query = translate_text(query, target_lang="ne")

            # 🔹 Retrieve relevant chunks using FAISS
            docs = vector_store.similarity_search(query=nepali_query, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])


            # 🔹 Updated concise prompt
            prompt = f"""
            Context:
            {context} # change to "context"

            User Question:
            {query}

            Instructions:
            - Provide a concise and direct response.
            - Use contract-style writing (brief, bullet points if applicable).
            - Avoid unnecessary explanations.
            - If additional information is needed, state "Not enough data in context."
            """

            # 🔹 Send context and query to OpenRouter API
            response = query_openrouter(prompt)

            # 🔹 Translate response to Nepali if user requested
            if lang_choice == "Nepali":
                response = translate_text(response, target_lang="ne")

            # ✅ Display AI-generated answer
            st.subheader("🤖 AI Response:")
            st.write(response)

          
if __name__ == "__main__":
    main()
