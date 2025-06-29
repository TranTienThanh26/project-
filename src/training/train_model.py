import tensorflow as tf

# Bước 1: Load dữ liệu từ thư mục
def load_dataset(folder_path):
    return tf.keras.preprocessing.image_dataset_from_directory(
        folder_path,
        image_size=(48, 48),
        color_mode="grayscale",  # ✅ Ảnh 1 kênh (grayscale)
        batch_size=32
    )

print("📁 Đang tải dữ liệu huấn luyện và kiểm thử...")
train_ds = load_dataset("model/fer2013/train")
test_ds = load_dataset("model/fer2013/test")

# Bước 2: Tạo mô hình CNN đơn giản
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(48, 48, 1)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(7, activation='softmax')  # FER2013 có 7 lớp cảm xúc
])

# Bước 3: Compile và huấn luyện mô hình
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("🚀 Bắt đầu huấn luyện mô hình...")
model.fit(train_ds, validation_data=test_ds, epochs=5)

# Bước 4: In mô hình
print("\n📊 Cấu trúc mô hình:")
model.summary()

# Bước 5: Lưu mô hình
print("\n💾 Đang lưu mô hình vào: model/emotion_model.h5")
model.save("model/emotion_model.h5")
print("✅ Đã lưu mô hình thành công.")
