import os
import cv2
from keras.models import load_model

# ✅ Danh sách cảm xúc
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# 📌 Load mô hình
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '../../model/emotion_model.h5')

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Không tìm thấy mô hình tại: {MODEL_PATH}")

model = load_model(MODEL_PATH)
print("✅ Đã load mô hình thành công!")

# 📷 Mở webcam MacBook
def find_macbook_camera():
    for index in [0, 1, 2]:
        cap = cv2.VideoCapture(index)
        if cap is not None and cap.read()[0]:
            print(f"🎥 Đang dùng webcam ở index {index}")
            return cap
        cap.release()
    print("❌ Không tìm thấy webcam hoạt động.")
    return None

cap = find_macbook_camera()
if cap is None:
    exit()

print("✅ Webcam đã mở.")

# 🔁 Hiển thị webcam
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Không đọc được frame.")
        break

    frame = cv2.flip(frame, 1)  # Lật ngang
    cv2.imshow('Webcam (Nhấn q để thoát)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 🧹 Dọn dẹp
cap.release()
cv2.destroyAllWindows()
