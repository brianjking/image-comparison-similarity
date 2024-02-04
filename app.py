import streamlit as st
from PIL import Image
import io
import boto3
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from efficientnet_pytorch import EfficientNet
import torchvision.transforms as transforms
import torch

# Initialize the text embeddings model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize the EfficientNet B7 model for image embeddings
efficientnet = EfficientNet.from_pretrained('efficientnet-b7')

def generate_image_embedding(image):
    input_size = EfficientNet.get_image_size('efficientnet-b7')  # Adjust for EfficientNet B7
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

# Define the function to extract text using Amazon Textract
def extract_text_with_textract(image_bytes):
    client = boto3.client('textract', region_name='us-east-1')  # Make sure to specify your region correctly
    response = client.analyze_document(Document={'Bytes': image_bytes}, FeatureTypes=['FORMS'])
    
    # Parsing the response to extract text
    text = ''
    for item in response['Blocks']:
        if item['BlockType'] == 'LINE':
            text += item['Text'] + '\n'
    return text

def main():
    st.title("Duplicate Document Detector")

    uploaded_file1 = st.file_uploader("Choose the first image...", type=["jpg", "png"])
    uploaded_file2 = st.file_uploader("Choose the second image...", type=["jpg", "png"])

    if uploaded_file1 and uploaded_file2:
        img_data1, img_data2 = uploaded_file1.read(), uploaded_file2.read()
        st.image([img_data1, img_data2], caption=['First Image', 'Second Image'], width=300)

        image1, image2 = Image.open(io.BytesIO(img_data1)), Image.open(io.BytesIO(img_data2))
        
        # Use Amazon Textract for OCR
        text1 = extract_text_with_textract(img_data1)
        text2 = extract_text_with_textract(img_data2)
        
        sentence_vector1, sentence_vector2 = model.encode([text1])[0], model.encode([text2])[0]
        image_embedding1, image_embedding2 = generate_image_embedding(image1), generate_image_embedding(image2)

        text_sim = cosine_similarity([sentence_vector1], [sentence_vector2])[0][0]
        image_sim = cosine_similarity([image_embedding1], [image_embedding2])[0][0]

        text_weight = 0.5
        image_weight = 0.5
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
