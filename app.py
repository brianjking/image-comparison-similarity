import streamlit as st
from PIL import Image
import io
import boto3
import numpy as np
import requests
import torchvision.transforms as transforms
import torch
from efficientnet_pytorch import EfficientNet
import faiss
from sklearn.metrics.pairwise import cosine_similarity
import os

# Constants for embedding dimensions
TEXT_EMBEDDING_DIM = 384  # all-MiniLM-L6 model output dimension
IMAGE_EMBEDDING_DIM = 1280  # EfficientNet-b1 output dimension

def init_or_load_faiss_index(index_path, embedding_dim):
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
    else:
        index = faiss.IndexFlatIP(embedding_dim)  # Use IndexFlatIP for cosine similarity
    return index
    
def normalize_embeddings(embeddings):
    if embeddings.ndim == 1:
        embeddings = np.expand_dims(embeddings, axis=0)
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    faiss.normalize_L2(embeddings)
    return embeddings

def add_to_faiss_index(index, embeddings):
    embeddings = normalize_embeddings(embeddings)
    index.add(embeddings)

def save_faiss_index(index, index_path):
    faiss.write_index(index, index_path)

def search_faiss_index(index, query_embedding, k=5):
    query_embedding = normalize_embeddings(np.expand_dims(query_embedding, axis=0))
    distances, indices = index.search(query_embedding, k)
    return distances, indices

def split_text(text, max_length=1024):
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

def generate_text_embedding(text):
    api_url = st.secrets["hugging_face_text_endpoint_url"]
    headers = {"Authorization": f"Bearer {st.secrets['hugging_face_api_key']}"}
    
    chunks = split_text(text)
    all_embeddings = []
    for chunk in chunks:
        payload = {"inputs": chunk}
        try:
            response = requests.post(api_url, headers=headers, json=payload)
            response.raise_for_status()
            chunk_embeddings = response.json()
            if isinstance(chunk_embeddings, list) and len(chunk_embeddings) > 0:
                all_embeddings.append(np.array(chunk_embeddings[0]))
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to process a text segment due to: {str(e)}")
            return None
    
    if all_embeddings:
        combined_embedding = np.mean(all_embeddings, axis=0)
        return combined_embedding
    else:
        st.error("Failed to generate text embeddings for any document segment.")
        return None

def generate_image_embedding(image):
    efficientnet = EfficientNet.from_pretrained('efficientnet-b1')
    input_size = EfficientNet.get_image_size('efficientnet-b1')
    transform = transforms.Compose([
        transforms.Resize(input_size, interpolation=Image.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = transform(image).unsqueeze(0)
    efficientnet.eval()
    with torch.no_grad():
        features = efficientnet.extract_features(image)
        embedding = torch.nn.functional.adaptive_avg_pool2d(features, 1).reshape(features.shape[0], -1)
    return embedding.numpy().flatten()

def extract_text(image_bytes):
    client = boto3.client(
        'textract',
        aws_access_key_id=st.secrets["aws_access_key_id"],
        aws_secret_access_key=st.secrets["aws_secret_access_key"],
        region_name=st.secrets["region_name"]
    )
    response = client.detect_document_text(Document={'Bytes': image_bytes})
    text_data = [item.get('Text', '') for item in response['Blocks'] if item['BlockType'] == 'LINE']
    return '\n'.join(text_data)

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
        
        add_to_faiss_index(text_index, sentence_vector1)
        add_to_faiss_index(image_index, image_embedding1)
        
        distances, indices = search_faiss_index(text_index, sentence_vector1, k=5)
        faiss_text_comparison_score = np.mean(1 - distances)  # Assuming distances are normalized
        
        # Display the top 5 matching hits from FAISS index
        st.write("Top 5 matching document indices:", indices[0])
        st.write("Corresponding distances:", distances[0])
        
        text_sim = cosine_similarity([sentence_vector1], [sentence_vector2])[0][0]
        image_sim = cosine_similarity([image_embedding1], [image_embedding2])[0][0]
        
        text_weight = 0.8
        image_weight = 0.2
        total_similarity = (text_sim * text_weight) + (image_sim * image_weight)
        
        with st.expander("Extracted Text from Images"):
            st.text_area("Text from First Image:", text1, height=150)
            st.text_area("Text from Second Image:", text2, height=150)

        st.write("Similarity Scores:")
        st.metric(label="Text Similarity", value=f"{text_sim*100:.2f}%")
        st.metric(label="Image Similarity", value=f"{image_sim*100:.2f}%")
        st.metric(label="Total Similarity", value=f"{total_similarity*100:.2f}%")
        st.metric(label="FAISS Text Comparison", value=f"{faiss_text_comparison_score*100:.2f}%")

        save_faiss_index(text_index, "text_index.faiss")
        save_faiss_index(image_index, "image_index.faiss")

if __name__ == "__main__":
    main()
