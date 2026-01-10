import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)
from PIL import Image

def load_model():
    model = MobileNetV2(weights="imagenet")
    return model

def preprocess_image(image):
    img = np.array(image)
    # 224x224 is the size expected by mobile net
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    # Expand the shape of an array.
    # Insert a new axis that will appear at the axis position in the expanded array shape.
    # Taking a single image and converting into a format that would represent multiple image
    # Since the model expects multiple images and we only pass one, we need to do this conversion
    # basically doing [img]
    img = np.expand_dims(img, axis=0)
    return img

def classify_image(model, image):
    try:
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        # take the numeric predictions and convert to string labels
        # take the top 3 predictions of the first image [0]
        decoded_predictions = decode_predictions(predictions, top=3)[0]
        return decoded_predictions
    except Exception as e:
        st.error(f"Error classifying image: {str(e)}")
        return None
    
def main():
    st.set_page_config(page_title="AI Image Classifier", page_icon="🌉", layout="centered")
    st.title("AI Image Classifier")
    st.write("Upload an image and let AI tell you what's in it!")
    
    @st.cache_resource
    def load_cached_model():
        return load_model()
    
    model = load_cached_model()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
    if uploaded_file is not None:
        image = st.image(
            uploaded_file, caption="Uploaded image", use_container_width=True
        )
        btn = st.button("Classify image")

        if btn:
            with st.spinner("In progress..."):
                image = Image.open(uploaded_file)
                predictions = classify_image(model, image)

                if predictions:
                    st.subheader("Preditions")
                    for _, label, score in predictions:
                        st.write(f"**{label}**: {score:.2%}")

if __name__ == "__main__":
    main()