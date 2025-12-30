# Mini NLP Chatbot with Retrieval-Augmented Generation (RAG)

A simple and lightweight chatbot that can answer questions from a text or PDF document using **retrieval-augmented generation (RAG)**. This app is built with **Python**, **Sentence Transformers**, **Scikit-learn**, and **Gradio**, and works fully offline once the model and text are loaded.


## Features

- **Upload text or PDF dynamically** – Users can upload any text file or PDF and get answers based on the content.  
- **Offline embeddings** – Embeddings are computed using a local Sentence Transformer model (`all-MiniLM-L6-v2`).  
- **Improved answer relevance** – Uses sentence-level retrieval for more focused responses.  
- **Simple and interactive UI** – Powered by Gradio with clear instructions and responsive design.  
- **Cross-platform compatibility** – Works on local machines, Kaggle, or Hugging Face Spaces.


## Usage

Upload a text or PDF file in the interface.

Type a question related to the uploaded document.

Click "Get Answer" to see the most relevant sentence(s) from the document.


