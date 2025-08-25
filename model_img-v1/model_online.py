import streamlit as st
from model_CNN_2 import getModelObject,predToClass
from PIL import Image
from torchvision.transforms import transforms
model = getModelObject()
transformation = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])
# Streamlit UI code
st.title('Welcome to Animal Image Classifier(V1)')
st.text("Predict Upto 90 Different Animals")
st.badge(label='Accuracy 70%',color='green')
uploaded_file = st.file_uploader("Choose an image...", type=["png","jpg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")
    image_tensor = transformation(image)
    logits = model(image_tensor.unsqueeze(0))
    prediction = predToClass(logits)
    st.markdown(f"**Prediction:** {prediction}")
