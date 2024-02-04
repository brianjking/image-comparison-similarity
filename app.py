import streamlit as st
from PIL import Image
import io
import pytesseract
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torchvision.transforms as transforms
import torch
from efficientnet_pytorch import EfficientNet

# Initialize the text embeddings model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize the EfficientNet B3 model for image embeddings
efficientnet = EfficientNet.from_pretrained('efficientnet-b7')

def generate_image_embedding(image):
    # Adjust preprocessing for EfficientNet B3's expected input dimensions
    input_size = EfficientNet.get_image_size('efficientnet-b7')
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
        # Extract features
        features = efficientnet.extract_features(image)
        # Pooling and flattening for embedding
        embedding = torch.nn.functional.adaptive_avg_pool2d(features, 1).reshape(features.shape[0], -1)
    return embedding.numpy().flatten()

def main():
    st.title("Duplicate Document Detector")

    # Upload images
    uploaded_file1 = st.file_uploader("Choose the first image...", type=["jpg", "png"])
    uploaded_file2 = st.file_uploader("Choose the second image...", type=["jpg", "png"])

    if uploaded_file1 and uploaded_file2:
        # Display images
        img_data1, img_data2 = uploaded_file1.read(), uploaded_file2.read()
        st.image([img_data1, img_data2], caption=['First Image', 'Second Image'], width=300)

        # Process images: Convert, OCR, Embeddings
        image1, image2 = Image.open(io.BytesIO(img_data1)), Image.open(io.BytesIO(img_data2))
        text1, text2 = pytesseract.image_to_string(image1), pytesseract.image_to_string(image2)
        sentence_vector1, sentence_vector2 = model.encode([text1])[0], model.encode([text2])[0]
        image_embedding1, image_embedding2 = generate_image_embedding(image1), generate_image_embedding(image2)

        # Calculate similarity scores
        text_sim = cosine_similarity([sentence_vector1], [sentence_vector2])[0][0]
        image_sim = cosine_similarity([image_embedding1], [image_embedding2])[0][0]

        # Weighted combination of scores
        text_weight = 0.9  # Adjust these weights as needed
        image_weight = 0.1
        total_similarity = (text_sim * text_weight) + (image_sim * image_weight)

        # Display OCR'ed text
        with st.expander("Extracted Text from Images"):
            st.text_area("Text from First Image:", text1, height=150)
            st.text_area("Text from Second Image:", text2, height=150)

        # Display similarity scores
        st.write("Similarity Scores:")
        st.metric(label="Text Similarity", value=f"{text_sim*100:.2f}%")
        st.metric(label="Image Similarity", value=f"{image_sim*100:.2f}%")
        st.metric(label="Total Similarity", value=f"{total_similarity*100:.2f}%")

if __name__ == "__main__":
    main()
