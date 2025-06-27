import cv2

def main():
    print("Khởi động ứng dụng nhận diện cảm xúc...")
    
    # Mở webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không thể mở webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không đọc được khung hình")
            break

        # Hiển thị video
        cv2.imshow("Webcam - Nhan dien cam xuc", frame)

        # Thoát bằng phím 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
