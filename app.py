import streamlit as st
from PIL import Image
import io
import boto3
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
import torchvision.transforms as transforms
import torch
from transformers import CLIPProcessor, CLIPModel

# Initialize CLIP model and processor globally
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def split_text(text, max_length=1024):
    """Splits the text into chunks that are at most `max_length` characters long."""
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

def generate_clip_image_embedding(image):
    inputs = clip_processor(images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    image_embeddings = outputs.image_embeds
    return image_embeddings.detach().numpy().flatten()

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
        
        if sentence_vector1 is None or sentence_vector2 is None:
            st.error("Failed to generate embeddings.")
            return
        
        image_embedding1 = generate_clip_image_embedding(image1)
        image_embedding2 = generate_clip_image_embedding(image2)

        text_sim = cosine_similarity([sentence_vector1], [sentence_vector2])[0][0]
        image_sim = cosine_similarity([image_embedding1], [image_embedding2])[0][0]

        text_weight = 0.9
        image_weight = 0.1
        total_similarity = (text_sim * text_weight) + (image_sim * image_weight)

        with st.expander("Extracted Text from Images"):
            st.text_area("Text from First Image:", text1, height=150)
            st.text_area("Text from Second Image:", text2, height=150)

        st.write("Similarity Scores:")
        st.metric(label="Text Similarity", value=f"{text_sim*100:.2f}%")
        st.metric(label="Image Similarity", value=f"{image_sim*100:.2f}%")
        st.metric(label="Total Similarity", value=f"{total_similarity*100:.2f}%")

if __name__ == "__main__":
    main()
