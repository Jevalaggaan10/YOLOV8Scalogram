import streamlit as st
import numpy as np
from PIL import Image
import time
from ultralytics import YOLO


def classify_scalogram(image,modelname):
    # Simulate a time-consuming task
    time.sleep(3)
    model = YOLO(modelname)  # load a custom model

    results = model(image)  # predict on an image

    names_dict = results[0].names

    probs = results[0].probs.data.tolist()
    confidence_score = probs[np.argmax(probs)]
    #print(names_dict)
    #print(probs)

    #print(names_dict[np.argmax(probs)])
    return f"{names_dict[np.argmax(probs)]} {round(confidence_score * 100, 2)}% (Confidence Score: {probs})"



st.header("Scalogram Epilepsy Classifier(YOLOV8)")

uploaded_file = st.file_uploader("Upload a Scalogram", type=["png"])
if uploaded_file is not None:
    if uploaded_file.name.lower().endswith(".png"):
        st.success("Uploaded file is a PNG image!")
        # image_bytes = uploaded_file.read()
        # image_array = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
        # Display the image using Streamlit
        # st.image(image_array, caption="Uploaded PNG Image", use_column_width=True)
        # print(uploaded_file)
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded PNG Image", use_column_width=True)
        if st.button(label="Classify Scalogram", type="primary"):
            with st.spinner("Classifying..."):
                result = classify_scalogram(image,'./runs/classify/train2/weights/last.pt')
                st.success(f"Classification result: {result}")
    else:
        st.warning("Please upload a valid PNG file.")
