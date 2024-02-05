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

def generate_text_embedding(text):
    # Use Hugging Face API for text embeddings
    api_url = st.secrets["hugging_face_text_endpoint_url"]
    headers = {"Authorization": f"Bearer {st.secrets['hugging_face_api_key']}"}
    payload = {"inputs": text}
    
    response = requests.post(api_url, headers=headers, json=payload)
    response.raise_for_status()  # Check for request errors
    embeddings = response.json()
    
    embedding = np.array(embeddings[0]) if isinstance(embeddings, list) and len(embeddings) > 0 else None
    return embedding

def generate_image_embedding(image):
    # Local processing for image embeddings using EfficientNet-B7
    efficientnet = EfficientNet.from_pretrained('efficientnet-b1')
    input_size = EfficientNet.get_image_size('efficientnet-b1')
    transform = transforms.Compose([
        transforms.Resize(input_size, interpolation=Image.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if image.mode != 'RGB':
        image = image.convert('RGB')  # Convert RGBA to RGB
    image = transform(image).unsqueeze(0)
    efficientnet.eval()
    with torch.no_grad():
        features = efficientnet.extract_features(image)
        embedding = torch.nn.functional.adaptive_avg_pool2d(features, 1).reshape(features.shape[0], -1)
    return embedding.numpy().flatten()

def extract_text_with_layout_analysis(image_bytes):
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
        
        sentence_vector1 = generate_text_embedding(text1)
        sentence_vector2 = generate_text_embedding(text2)
        image_embedding1 = generate_image_embedding(image1)
        image_embedding2 = generate_image_embedding(image2)

        text_sim = cosine_similarity([sentence_vector1], [sentence_vector2])[0][0]
        image_sim = cosine_similarity([image_embedding1], [image_embedding2])[0][0]

        text_weight = 0.9
        image_weight = 0.1
        total_similarity = (text_sim * text_weight) + (image_sim * image_weight)

        with st.expander("Extracted Text and Layout from Images"):
            st.text_area("Text and Layout from First Image:", str(layout_data1), height=150)
            st.text_area("Text and Layout from Second Image:", str(layout_data2), height=150)

        st.write("Similarity Scores:")
        st.metric(label="Text Similarity", value=f"{text_sim*100:.2f}%")
        st.metric(label="Image Similarity", value=f"{image_sim*100:.2f}%")
        st.metric(label="Total Similarity", value=f"{total_similarity*100:.2f}%")

if __name__ == "__main__":
    main()
