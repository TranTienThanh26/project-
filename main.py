import cv2
import os
from datetime import datetime

# Tải mô hình Haar cascade detect face
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Tạo thư mục lưu ảnh khuôn mặt
output_dir = "faces"
os.makedirs(output_dir, exist_ok=True)

# Mở webcam (0 là webcam mặc định)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không thể mở webcam.")
    exit()

face_id = 0  # Đếm số khuôn mặt đã lưu

print("🔴 Đang mở webcam... Nhấn 'q' để thoát.")

while True:
    # Đọc frame từ webcam
    ret, frame = cap.read()
    if not ret:
        print("❌ Không nhận được khung hình từ webcam.")
        break

    # Chuyển sang ảnh xám để detect
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect khuôn mặt
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Vẽ khung xanh quanh khuôn mặt
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Cắt và resize ảnh mặt về 48x48 (chuẩn FER2013)
        face_img = gray[y:y+h, x:x+w]
        resized_face = cv2.resize(face_img, (48, 48))

        # Đặt tên file kèm thời gian
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(output_dir, f"face_{face_id}_{timestamp}.png")

        # Lưu ảnh
        cv2.imwrite(file_path, resized_face)
        print(f"✅ Đã lưu khuôn mặt #{face_id} → {file_path}")
        face_id += 1

    # Hiển thị frame realtime
    cv2.imshow("Webcam - Detect Face", frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng camera và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()
print("🟢 Đã tắt webcam và kết thúc chương trình.")
