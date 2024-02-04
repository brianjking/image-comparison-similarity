import streamlit as st
from PIL import Image
import io
import pytesseract
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torchvision.models as models
import torchvision.transforms as transforms
import torch

# Text embeddings model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Pretrained model and transform for image embeddings
resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def generate_image_embedding(image):
    # Convert RGBA images to RGB
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    image = transform(image).unsqueeze(0)
    resnet.eval()
    with torch.no_grad():
        embedding = resnet(image).numpy().flatten()
    return embedding

def main():
    st.title("Duplicate Document Detector")
    sentence_vector1 = None
    sentence_vector2 = None
    image_embedding1 = None
    image_embedding2 = None

    uploaded_file1 = st.file_uploader("Choose the first image...", type=["jpg", "png"])
    uploaded_file2 = st.file_uploader("Choose the second image...", type=["jpg", "png"])

    if uploaded_file1 is not None:
        img_data1 = uploaded_file1.read()
        st.image(img_data1, caption='First Image')
        image1 = Image.open(io.BytesIO(img_data1))
        text1 = pytesseract.image_to_string(image1)
        st.write('Extracted Text from First Image:')
        st.write(text1)
        sentence_vector1 = model.encode([text1])[0]
        st.write('Text Embedding for First Image:')
        st.write(sentence_vector1)
        image_embedding1 = generate_image_embedding(image1)
        st.write('Image Embedding for First Image:')
        st.write(image_embedding1)

    if uploaded_file2 is not None:
        img_data2 = uploaded_file2.read()
        st.image(img_data2, caption='Second Image')
        image2 = Image.open(io.BytesIO(img_data2))
        text2 = pytesseract.image_to_string(image2)
        st.write('Extracted Text from Second Image:')
        st.write(text2)
        sentence_vector2 = model.encode([text2])[0]
        st.write('Text Embedding for Second Image:')
        st.write(sentence_vector2)
        image_embedding2 = generate_image_embedding(image2)
        st.write('Image Embedding for Second Image:')
        st.write(image_embedding2)

    if sentence_vector1 is not None and sentence_vector2 is not None:
        sim = cosine_similarity([sentence_vector1], [sentence_vector2])[0][0]
        st.write(f'Text similarity between images: {sim*100:.2f}%')

    if image_embedding1 is not None and image_embedding2 is not None:
        sim = cosine_similarity([image_embedding1], [image_embedding2])[0][0]
        st.write(f'Image similarity between images: {sim*100:.2f}%')

if __name__ == "__main__":
    main()
