import streamlit as st
import pickle
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import models

class_index_to_name = {
        1: 'Apple___Apple_scab',
        2: 'Apple___Black_rot',
        3: 'Apple___Cedar_apple_rust',
        4: 'Apple___healthy',
        5: 'Blueberry___healthy',
        6: 'Cherry_(including_sour)___Powdery_mildew',
        7: 'Cherry_(including_sour)___healthy',
        8: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
        9: 'Corn_(maize)___Common_rust_',
        10: 'Corn_(maize)___Northern_Leaf_Blight',
        11: 'Corn_(maize)___healthy',
        12: 'Grape___Black_rot',
        13: 'Grape___Esca_(Black_Measles)',
        14: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
        15: 'Grape___healthy',
        16: 'Orange___Haunglongbing_(Citrus_greening)',
        17: 'Peach___Bacterial_spot',
        18: 'Peach___healthy',
        19: 'Pepper,_bell___Bacterial_spot',
        20: 'Pepper,_bell___healthy',
        21: 'Potato___Early_blight',
        22: 'Potato___Late_blight',
        23: 'Potato___healthy',
        24: 'Raspberry___healthy',
        25: 'Soybean___healthy',
        26: 'Squash___Powdery_mildew',
        27: 'Strawberry___Leaf_scorch',
        28: 'Strawberry___healthy',
        29: 'Tomato___Bacterial_spot',
        30: 'Tomato___Early_blight',
        31: 'Tomato___Late_blight',
        32: 'Tomato___Leaf_Mold',
        33: 'Tomato___Septoria_leaf_spot',
        34: 'Tomato___Spider_mites Two-spotted_spider_mite',
        35: 'Tomato___Target_Spot',
        36: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
        37: 'Tomato___Tomato_mosaic_virus',
        38: 'Tomato___healthy'
    }

class CnnModel(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.resnet_layers = nn.Sequential(*list(resnet.children())[:-1])
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 38)
        
    def forward(self, xb):
        x = self.resnet_layers(xb)
        x = x.view(x.size(0), -1)
        out = self.fc1(x)
        out = self.fc2(out)
        return out

# Load the trained model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

# Set page configurations
st.set_page_config(
    page_title="🌱 Easy Farming",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="auto",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-image: linear-gradient(to bottom, #00843D, #00CBA4);
        color: white;
    }
    .css-1i3mj7p {
        background-color: #00CBA4;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Custom styling for title
st.title(
    "Easy Farming",
    anchor="center",
)
st.subheader("🌿 Your Digital Farming Companion")

# Sidebar
with st.sidebar:
    st.image("farm_icon.png", width=100)
    st.markdown("## 🌱 Navigate")
    selected_page = st.selectbox(
        "",
        ("Home", "Crop Recommendation", "Plant Disease Identification"),
    )

if selected_page == "Home":
    st.markdown("## 🏡 Welcome to Easy Farming!")
    st.write(
        "Your journey to smart agriculture begins here. Select a function from the sidebar to get started."
    )
    st.image("farm_image.jpg", width=500)

elif selected_page == "Crop Recommendation":
    st.markdown("## 🌾 Crop Recommendation System")
    st.write(
        "Need guidance on choosing the right crop? We've got you covered. Provide some soil and weather information, and let us recommend the best crop for you!"
    )
    st.header("Enter Values for Crop Recommendation")
    
    nitrogen = st.number_input("Nitrogen in Soil (kg/ha)")
    phosphorous = st.number_input("Phosphorous in Soil (kg/ha)")
    potassium = st.number_input("Potassium in Soil (kg/ha)")
    temperature = st.number_input("Average Temperature (°C)")
    humidity = st.number_input("Average Humidity (%)")
    ph_value = st.number_input("Average pH Value")
    rainfall = st.number_input("Rainfall Amount (mm)")

    if st.button("Predict Crop"):
        new_data = [nitrogen, phosphorous, potassium, temperature, humidity, ph_value, rainfall]
        prediction = loaded_model.predict([new_data])
        st.success(f"Predicted Crop Class: **{prediction[0]}**")
    

elif selected_page == "Plant Disease Identification":
    st.markdown("## 🍃 Plant Disease Identification System")
    st.write(
        "Detecting plant diseases is now easier than ever. Simply upload an image of your plant, and we'll identify the disease affecting it."
    )
    st.header("Upload Image for Disease Identification")
    uploaded_disease_image = st.file_uploader("Upload an image below", type=["jpg", "png", "jpeg"])
    # Add more content here for the plant disease identification section
    if uploaded_disease_image is not None:
        disease_image = Image.open(uploaded_disease_image).convert('RGB')
        st.image(disease_image, caption="Uploaded Disease Image", use_column_width=True)

    if st.button("Identify Disease"):
        model = CnnModel()
        model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
        model.eval()
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
           
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
        image = transform(disease_image)
        image = image.unsqueeze(0)

        with torch.no_grad():
            output = model(image)
            predicted_class = torch.argmax(output, dim=1).item()
        predicted_class_name = class_index_to_name.get(predicted_class, 'Unknown')
        
        st.success("Disease identified!")
        st.markdown('<p style="text-align: center; font-size: 24px; color: #FFFFFF;">Predicted Disease</p>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; font-size: 48px; color: #FF5733;"><strong>{}</strong></p>'.format(predicted_class_name), unsafe_allow_html=True)


# Footer
st.markdown(
    "---\nMade with ❤️ by [Ajay](your_github_repo_url)", unsafe_allow_html=True
)
