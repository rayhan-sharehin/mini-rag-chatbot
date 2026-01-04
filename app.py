
import os
import re
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, models
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr
import PyPDF2

# Force CPU & offline mode
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_default_device("cpu")
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# Load offline SentenceTransformer
MODEL_PATH = "all-MiniLM-L6-v2"  # MY offline model folder here

word_embedding_model = models.Transformer(MODEL_PATH)
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True
)
embedder = SentenceTransformer(modules=[word_embedding_model, pooling_model], device="cpu")
print("Model loaded successfully ")

# Helper functions
def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if len(s) > 10]

def embed_texts(texts):
    return embedder.encode(texts, convert_to_numpy=True)

# Global variables
all_sentences = []
sentence_embeddings = None
all_chunks = []
chunk_embeddings = None

# File processing
def process_file(file):
    global all_sentences, sentence_embeddings, all_chunks, chunk_embeddings
    
    if file.name.endswith(".txt"):
        with open(file.name, "r", encoding="utf-8") as f:
            text = f.read()
    elif file.name.endswith(".pdf"):
        text = ""
        reader = PyPDF2.PdfReader(file.name)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    else:
        return "Only .txt or .pdf files are supported"
    
    # Split into chunks
    all_chunks = split_text(text)
    chunk_embeddings = embed_texts(all_chunks)
    
    # Split into sentences
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    sentences = []
    for para in paragraphs:
        sentences.extend(split_into_sentences(para))
    
    all_sentences = sentences
    sentence_embeddings = embed_texts(all_sentences)
    
    return f"File processed ✅ Total chunks: {len(all_chunks)}, Total sentences: {len(all_sentences)}"

# Query function (top 3)
def retrieve_answer(query, top_k=3):
    if chunk_embeddings is None or sentence_embeddings is None:
        return "Upload a file first!"
    
    query_emb = embed_texts([query])
    
    # Step 1: retrieve top 3 chunks
    chunk_sims = cosine_similarity(query_emb, chunk_embeddings)[0]
    top_chunk_indices = np.argsort(chunk_sims)[-3:][::-1]
    
    # Step 2: collect sentences from top chunks
    candidate_sentences = []
    candidate_embeddings = []
    for idx in top_chunk_indices:
        chunk_text = all_chunks[idx]
        sents = split_into_sentences(chunk_text)
        candidate_sentences.extend(sents)
    
    candidate_embeddings = embed_texts(candidate_sentences)
    
    # Step 3: compute similarity with query
    sims = cosine_similarity(query_emb, candidate_embeddings)[0]
    top_indices = np.argsort(sims)[-top_k:][::-1]
    
    results = []
    for i, idx in enumerate(top_indices):
        results.append(f"**Result {i+1}:** {candidate_sentences[idx]}")
    
    return "\n\n".join(results)


# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align:center'>📄 Rayhan's Mini NLP Chatbot</h1>", elem_id="header")
    gr.Markdown(
        "<p style='text-align:center'>This is a RAG-NLP based fully responsive document QA system. Upload a text or PDF file, then ask questions about its content. "
        "The chatbot retrieves the most relevant sentences from the document.</p>"
    )
    
    with gr.Row():
        file_input = gr.File(label="Upload a text or PDF file here", file_types=[".txt", ".pdf"])
        upload_btn = gr.Button("Process File", elem_id="upload-btn")
    
    file_status = gr.Textbox(label="File Status")
    
    with gr.Row():
        query_input = gr.Textbox(label="Ask a question from the uploaded file!")
        query_btn = gr.Button("Get Answer", elem_id="answer-btn")
    
    answer_output = gr.Markdown(label="Answer")
    
    upload_btn.click(process_file, inputs=file_input, outputs=file_status)
    query_btn.click(retrieve_answer, inputs=query_input, outputs=answer_output)

# Custom CSS for orange button
custom_css = """
#answer-btn button {
    background-color: orange !important;
    color: white !important;
    font-weight: bold !important;
    font-size: 16px !important;
}
"""

demo.launch(server_name="0.0.0.0", server_port=7860, css=custom_css)
