import io
import chromadb
from chromadb.config import Settings
import ollama
import pypdf
from PIL import Image
import os
import torch
from transformers import AutoModel, AutoTokenizer
import tempfile

# Initialize global clients
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection_name = "pdf_rag_collection"

# Global OCR Model Placeholder
ocr_model = None
ocr_tokenizer = None

def load_ocr_model():
    global ocr_model, ocr_tokenizer
    if ocr_model is None:
        try:
            model_name = "deepseek-ai/DeepSeek-OCR"
            ocr_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            ocr_model = AutoModel.from_pretrained(model_name, trust_remote_code=True, use_safetensors=True)
            
            # Check for CUDA
            if torch.cuda.is_available():
                ocr_model = ocr_model.cuda().to(torch.bfloat16)
                print("DeepSeek-OCR loaded on GPU.")
            else:
                print("DeepSeek-OCR loaded on CPU (Warning: Slow).")
                
            ocr_model.eval()
        except Exception as e:
            print(f"Failed to load DeepSeek-OCR: {e}")
            raise e

def get_chroma_collection():
    return chroma_client.get_or_create_collection(name=collection_name)

def process_pdf(file_bytes, use_ocr=False, vision_model='deepseek-ai/DeepSeek-OCR'):
    """
    Extracts text from PDF.
    If use_ocr is True, converts to image and uses local DeepSeek-OCR (transformers).
    Otherwise uses pypdf.
    """
    text_content = ""
    
    if not use_ocr:
        # Standard pypdf extraction
        try:
            pdf_reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            for page in pdf_reader.pages:
                text_content += page.extract_text() + "\n"
        except Exception as e:
            return f"Error reading PDF: {e}"
    else:
        # OCR Path using Transformers DeepSeek-OCR
        try:
            # Ensure model is loaded
            load_ocr_model()
            
            from pdf2image import convert_from_bytes
            import sys
            import os

            # Attempt to find poppler in Conda environment
            poppler_path = None
            if sys.platform.startswith("win"):
                conda_poppler_path = os.path.join(sys.prefix, "Library", "bin")
                if os.path.exists(os.path.join(conda_poppler_path, "pdftoppm.exe")):
                    poppler_path = conda_poppler_path
            
            if poppler_path:
                images = convert_from_bytes(file_bytes, poppler_path=poppler_path)
            else:
                images = convert_from_bytes(file_bytes)
            
            # Temporary directory for image processing
            with tempfile.TemporaryDirectory() as temp_dir:
                for i, image in enumerate(images):
                    image_path = os.path.join(temp_dir, f"page_{i}.jpg")
                    image.save(image_path, format='JPEG')
                    
                    try:
                        # DeepSeek-OCR Inference
                        prompt = "<image>\n<|grounding|>Convert the document to markdown. "
                        
                        res = ocr_model.infer(
                            ocr_tokenizer,
                            prompt=prompt,
                            image_file=image_path,
                            output_path=temp_dir, # Model might write debug outputs here
                            base_size=1024,
                            image_size=640,
                            crop_mode=True,
                            save_results=False, 
                            test_compress=True 
                        )
                        
                        text_content += f"\n[Page {i+1}]\n{res}\n"
                        
                    except Exception as e:
                        text_content += f"\n[Page {i+1} Error: {str(e)}]\n"

        except ImportError:
            return "Error: pdf2image not installed or poppler not found. Please install poppler."
        except Exception as e:
             return f"OCR Error: {e}\n(Ensure CUDA is available if using DeepSeek-OCR)"

    if not text_content.strip():
        return "No text could be extracted."
        
    return text_content

def index_document(text, embedding_model='nomic-embed-text'):
    """
    Embeds text using Ollama (nomic-embed-text) and stores in ChromaDB.
    """
    
    # Simple chunking
    chunk_size = 500
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    # Create IDs
    ids = [f"id_{i}" for i in range(len(chunks))]
    
    # Ollama Embedding
    embeddings = []
    try:
        for chunk in chunks:
            response = ollama.embeddings(model=embedding_model, prompt=chunk)
            embeddings.append(response['embedding'])
    except Exception as e:
        raise Exception(f"Ollama Embedding Error: {e}. Ensure model '{embedding_model}' is pulled.")
    
    # Store in Chroma
    collection = get_chroma_collection()
    # Reset for demo purposes
    chroma_client.delete_collection(collection_name)
    collection = chroma_client.get_or_create_collection(collection_name)
    
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids
    )

def query_rag(question, chat_model='deepseek-r1', embedding_model='nomic-embed-text'):
    """
    Retrieves context and answers question using Ollama.
    """
    collection = get_chroma_collection()
    
    # Embed question
    try:
        response = ollama.embeddings(model=embedding_model, prompt=question)
        query_embedding = response['embedding']
    except Exception as e:
        return f"Embedding Error: {e}"
    
    # Retrieve
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    
    if not results['documents'][0]:
        return "No relevant context found."

    context = "\n".join(results['documents'][0])
    
    # Generate Answer
    try:
        response = ollama.chat(
            model=chat_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer the user's question. If the context doesn't contain the answer, say so."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ]
        )
        return response['message']['content']
    except Exception as e:
        return f"Generation Error: {e}. Ensure model '{chat_model}' is pulled."
