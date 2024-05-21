import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# MobileNetV2 모델 로드
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# 이미지 전처리 함수
def preprocess_image(image):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img

# 꽃 이름을 예측하는 함수
def predict_flower(image):
    img = preprocess_image(image)
    predictions = model.predict(img)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)
    return decoded_predictions[0][0][1]

def streamlit_app():
    # Streamlit 웹 페이지 설정
    st.title('Flower Name Prediction')
    st.write('Upload an image or use the camera to capture an image of a flower and the app will predict its name.')

    # 파일 업로드
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # 카메라 입력
    camera_input = st.camera_input("Take a picture...")

    # 이미지를 표시하고 예측
    if uploaded_file is not None or camera_input is not None:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
        else:
            image = Image.open(camera_input)

        st.image(image, caption='Uploaded Image', use_column_width=True)

        # 예측 버튼
        if st.button('Predict'):
            flower_name = predict_flower(image)
            st.write(f'The flower in the image is: {flower_name}')

if __name__ == "__main__":
    streamlit_app()
