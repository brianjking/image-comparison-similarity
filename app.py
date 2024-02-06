import streamlit as st
from PIL import Image
import io
import boto3
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
import torchvision.transforms as transforms
import torch
from efficientnet_pytorch import EfficientNet
import faiss
import os

# Constants for embedding dimensions
TEXT_EMBEDDING_DIM = 384  # all-MiniLM-L6 model output dimension
IMAGE_EMBEDDING_DIM = 1280  # EfficientNet-b1 output dimension

def init_or_load_faiss_index(index_path, embedding_dim):
    if os.path.exists(index_path):
        return faiss.read_index(index_path)
    else:
        index = faiss.IndexFlatL2(embedding_dim)
        return index

def add_to_faiss_index(index, embeddings):
    if embeddings.ndim == 1:
        embeddings = np.expand_dims(embeddings, axis=0)
    index.add(embeddings.astype('float32'))

def save_faiss_index(index, index_path):
    faiss.write_index(index, index_path)

def get_closest_embeddings(index, query_embedding, k=5):
    query_embedding = np.expand_dims(query_embedding, axis=0).astype('float32')
    _, I = index.search(query_embedding, k)  # Ignore distances, just get indices of closest
    return I

# Assume other functions (split_text, generate_text_embedding, generate_image_embedding, extract_text) are defined here

def calculate_similarity_index(index, embedding):
    # Retrieve the top 5 nearest embeddings' indices
    indices = get_closest_embeddings(index, embedding)
    # For simplicity, just return the indices. In practice, you might calculate similarity scores based on these.
    return indices

def main():
    st.title("Duplicate Document Detector")
    
    text_index = init_or_load_faiss_index("text_index.faiss", TEXT_EMBEDDING_DIM)
    image_index = init_or_load_faiss_index("image_index.faiss", IMAGE_EMBEDDING_DIM)
    
    uploaded_file1 = st.file_uploader("Choose the first image...", type=["jpg", "png"])
    uploaded_file2 = st.file_uploader("Choose the second image...", type=["jpg", "png"])

    if uploaded_file1 and uploaded_file2:
        img_data1, img_data2 = uploaded_file1.read(), uploaded_file2.read()
        st.image([img_data1, img_data2], caption=['First Image', 'Second Image'], width=300)
        
        image1, image2 = Image.open(io.BytesIO(img_data1)), Image.open(io.BytesIO(img_data2))
        
        text1 = extract_text(img_data1)
        text2 = extract_text(img_data2)
        sentence_vector1 = generate_text_embedding(text1)
        sentence_vector2 = generate_text_embedding(text2)
        
        image_embedding1 = generate_image_embedding(image1)
        image_embedding2 = generate_image_embedding(image2)
        
        # Add embeddings to FAISS index
        add_to_faiss_index(text_index, sentence_vector1)
        add_to_faiss_index(image_index, image_embedding1)
        
        # Calculate similarity with existing embeddings in the index
        nearest_text_indices = calculate_similarity_index(text_index, sentence_vector1)
        nearest_image_indices = calculate_similarity_index(image_index, image_embedding1)
        
        text_sim = cosine_similarity([sentence_vector1], [sentence_vector2])[0][0] if sentence_vector1 is not None and sentence_vector2 is not None else 0
        image_sim = cosine_similarity([image_embedding1], [image_embedding2])[0][0]
        
        text_weight = 0.8
        image_weight = 0.2
        total_similarity = (text_sim * text_weight) + (image_sim * image_weight)
        
        # Display extracted text and similarity scores
        with st.expander("Extracted Text from Images"):
            st.text_area("Text from First Image:", text1, height=150)
            st.text_area("Text from Second Image:", text2, height=150)

        st.write("Similarity Scores:")
        st.metric(label="Text Similarity", value=f"{text_sim*100:.2f}%")
        st.metric(label="Image Similarity", value=f"{image_sim*100:.2f}%")
        st.metric(label="Total Similarity", value=f"{total_similarity*100:.2f}%")

        # Display nearest vectors
        st.write(f"Nearest text indices for Image 1: {nearest_text_indices.flatten()}")
        st.write(f"Nearest image indices for Image 1: {nearest_image_indices.flatten()}")

        # Remember to add newly generated embeddings for image 2 and calculate/display all relevant metrics and nearest indices
        
        save_faiss_index(text_index, "text_index.faiss")
        save_faiss_index(image_index, "image_index.faiss")

if __name__ == "__main__":
    main()
