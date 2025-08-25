import streamlit as st
from torch import inference_mode
from PIL import Image
from torchvision.transforms import transforms
from model_CNN_2 import getModelObject,predToClass

    # Streamlit UI code
st.title('Welcome to Animal Image Classifier(V1)')
st.text("Predict Upto 90 Different Animals")
st.badge(label='Accuracy 70%',color='green')
if "model" not in st.session_state:
    try:
        st.session_state["model"],st.session_state["device"] = getModelObject()
    except Exception as e:
        st.exception(e) 
        st.stop()   
model = st.session_state["model"] 
transformation = transforms.Compose([
transforms.Resize((64,64)),
transforms.ToTensor()
])
uploaded_file = st.file_uploader("Choose an image...", type=["png","jpg"])
if uploaded_file:
    try:
        model = st.session_state.model
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image")
        image_fit = transformation(image).unsqueeze(0).to(st.session_state["device"])
        with inference_mode():
            logits = model(image_fit)
        prediction = predToClass(logits)
        st.markdown(f"**Prediction:**  {prediction}")
    except Exception as e:
        st.exception(e)
        