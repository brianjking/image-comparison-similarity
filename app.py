import streamlit as st
from PIL import Image
import io
import boto3
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
import torchvision.transforms as transforms
import torch

# Initialize the text embeddings model
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_image_embedding(image):
    # Convert PIL Image to byte array to send as binary data
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')  # Save image as JPEG to byte array
    img_byte_arr = img_byte_arr.getvalue()  # Get binary data
    
    # Set up headers with the Hugging Face API key from Streamlit secrets
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {st.secrets['hugging_face_api_key']}",
        "Content-Type": "image/jpeg"
    }
    
    # Use the provided endpoint URL from Streamlit secrets
    api_url = st.secrets["hugging_face_endpoint_url"]
    
    # Make the POST request to the Hugging Face Inference API
    response = requests.post(api_url, headers=headers, data=img_byte_arr)
    response.raise_for_status()  # Raises an exception for HTTP error responses
    
    # Extract the embedding from the response
    embedding = response.json()
    
    # Convert embedding to the required format (e.g., numpy array)
    embedding = np.array(embedding)
    
    return embedding.flatten()

def extract_text_with_layout_analysis(image_bytes):
    # Utilize AWS credentials from Streamlit secrets
    client = boto3.client(
        'textract',
        aws_access_key_id=st.secrets["aws_access_key_id"],
        aws_secret_access_key=st.secrets["aws_secret_access_key"],
        region_name=st.secrets["region_name"]
    )
    response = client.analyze_document(Document={'Bytes': image_bytes}, FeatureTypes=['TABLES', 'FORMS'])
    
    layout_data = []
    for item in response['Blocks']:
        if item['BlockType'] in ['LINE', 'WORD']:
            text = item.get('Text', '')
            bounding_box = item.get('Geometry', {}).get('BoundingBox', {})
            layout_data.append({'text': text, 'bounding_box': bounding_box})
    return layout_data

def main():
    st.title("Duplicate Document Detector")

    uploaded_file1 = st.file_uploader("Choose the first image...", type=["jpg", "png"])
    uploaded_file2 = st.file_uploader("Choose the second image...", type=["jpg", "png"])

    if uploaded_file1 and uploaded_file2:
        img_data1, img_data2 = uploaded_file1.read(), uploaded_file2.read()
        st.image([img_data1, img_data2], caption=['First Image', 'Second Image'], width=300)

        image1, image2 = Image.open(io.BytesIO(img_data1)), Image.open(io.BytesIO(img_data2))
        
        layout_data1 = extract_text_with_layout_analysis(img_data1)
        layout_data2 = extract_text_with_layout_analysis(img_data2)
        
        text1 = '\n'.join([item['text'] for item in layout_data1])
        text2 = '\n'.join([item['text'] for item in layout_data2])
        
        sentence_vector1, sentence_vector2 = model.encode([text1])[0], model.encode([text2])[0]
        image_embedding1, image_embedding2 = generate_image_embedding(image1), generate_image_embedding(image2)

        text_sim = cosine_similarity([sentence_vector1], [sentence_vector2])[0][0]
        image_sim = cosine_similarity([image_embedding1], [image_embedding2])[0][0]

        text_weight = 0.9
        image_weight = 0.1
        total_similarity = (text_sim * text_weight) + (image_sim * image_weight)

        with st.expander("Extracted Text and Layout from Images"):
            st.text_area("Text and Layout from First Image:", str(layout_data1), height=150)
            st.text_area("Text and Layout from Second Image:", str(layout_data2), height=150)

        st.write("Similarity Scores:")
        st.metric(label="Text Similarity (80%)", value=f"{text_sim*100:.2f}%")
        st.metric(label="Image Similarity (20%)", value=f"{image_sim*100:.2f}%")
        st.metric(label="Total Similarity", value=f"{total_similarity*100:.2f}%")

if __name__ == "__main__":
    main()
