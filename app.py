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
import time

def generate_text_embedding(text):
    api_url = st.secrets["hugging_face_text_endpoint_url"]
    headers = {"Authorization": f"Bearer {st.secrets['hugging_face_api_key']}"}
    # Assuming an average of 4 chars per token as a simple heuristic
    max_chars = 16384 * 4  # Adjust based on your observation or tokenization logic
    chunks = [text[i:i+max_chars] for i in range(0, len(text), max_chars)]
    
    embeddings = []
    for chunk in chunks:
        payload = {"inputs": chunk}
        try:
            response = requests.post(api_url, headers=headers, json=payload)
            response.raise_for_status()  # This will raise an exception for HTTP errors
            chunk_embeddings = response.json()
            if isinstance(chunk_embeddings, list) and len(chunk_embeddings) > 0:
                embeddings.extend(chunk_embeddings)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 413:
                st.error("Error: A document segment is too large to process. Please try shorter text.")
                return None
            else:
                st.error(f"An HTTP error occurred: {e}")
                return None
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            return None
    
    if embeddings:
        # Here you may adjust how to aggregate embeddings from chunks, e.g., averaging
        combined_embedding = np.mean(np.array(embeddings), axis=0)
        return combined_embedding
    else:
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
            st.error("Failed to generate text embeddings for comparison.")
            return
        
        image_embedding1 = generate_image_embedding(image1)
        image_embedding2 = generate_image_embedding(image2)

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
