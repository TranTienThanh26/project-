# File: predict_emotion.py

import tensorflow as tf
import numpy as np
import cv2

# Bản đồ nhãn cảm xúc
label_map = {
    0: "Angry", 1: "Disgust", 2: "Fear",
    3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"
}

# Load model một lần duy nhất
model = tf.keras.models.load_model("model/emotion_model.h5")

def predict_emotion(img):
    """
    Dự đoán cảm xúc từ ảnh BGR (ảnh OpenCV)
    Trả về: (label: int, label_name: str)
    """
    img = cv2.resize(img, (48, 48))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.shape[-1] == 3 else img
    img = img.reshape(1, 48, 48, 1).astype("float32") / 255.0

    # ✅ Hiển thị thanh tiến trình khi predict
    prediction = model.predict(img, verbose=1)
    label = np.argmax(prediction)
    return label, label_map[label]

# Cho phép chạy thử độc lập
if __name__ == "__main__":
    img = cv2.imread("test_sample.png")
    if img is None:
        print("❌ Không tìm thấy ảnh test_sample.png")
    else:
        label, emotion = predict_emotion(img)
        print(f"😊 Cảm xúc dự đoán: {emotion} (label {label})")
