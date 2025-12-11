import streamlit as st
import rag_engine
import os

st.set_page_config(page_title="DeepSeek & Nomic Local RAG (Ollama)", layout="wide")

st.title("DeepSeek OCR & Nomic Local RAG (Ollama)")

# Sidebar for Configuration
with st.sidebar:  
    st.header("Ollama Configuration")
    
    # Model Selection
    ocr_model = st.text_input("OCR/Vision Model", value="deepseek-ai/DeepSeek-OCR", help="Transformers model path.")
    embed_model = st.text_input("Embedding Model", value="nomic-embed-text", help="Model used for embeddings.")
    chat_model = st.text_input("Chat Model", value="deepseek-r1", help="Model used for answering questions.")
    
    use_ocr = st.checkbox("Enable Local OCR (Vision)", value=False, help="Requires a vision model like 'llava' or 'moondream' pulled in Ollama, and 'poppler' installed on system.")
    
    if st.button("Clear Knowledge Base"):
        try:
            rag_engine.chroma_client.delete_collection(rag_engine.collection_name)
            st.success("Cleared!")
        except:
            st.info("Already empty.")

    st.info(f"Using models:\n- OCR: {ocr_model}\n- Embed: {embed_model}\n- Chat: {chat_model}")

# Main Interface
uploaded_file = st.file_uploader("Upload a PDF Document", type=["pdf"])

if uploaded_file is not None:
    # Process Button
    if st.button("Process Document"):
        with st.spinner(f"Processing... Extracting Text ({ocr_model if use_ocr else 'PDF Text'})..."):
            file_bytes = uploaded_file.read()
            
            # Step 1: Text Extraction/OCR
            try:
                # Pass file bytes to engine
                text = rag_engine.process_pdf(file_bytes, use_ocr=use_ocr, vision_model=ocr_model)
                
                # Check for extraction failure
                if "Error" in text or not text.strip():
                    st.error(text)
                else:
                    st.info(f"Extracted {len(text)} characters.")
                    with st.expander("View Extracted Text"):
                        st.write(text[:1000] + "...")
                        
                    # Step 2: Embedding & Indexing
                    with st.spinner(f"Embedding with {embed_model}..."):
                        rag_engine.index_document(text, embedding_model=embed_model)
                    st.success("Document processed and embedded successfully!")
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Chat Interface
st.divider()
st.header("Ask Questions")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask something about the uploaded PDF..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(f"Thinking with {chat_model}..."):
            try:
                response = rag_engine.query_rag(prompt, chat_model=chat_model, embedding_model=embed_model)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error generating response: {e}")
