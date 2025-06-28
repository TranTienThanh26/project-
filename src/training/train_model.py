import tensorflow as tf
from prepare_dataset import load_dataset

# Bước 1: Load dữ liệu
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

# Bước 3: Compile & Train
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds, validation_data=test_ds, epochs=5)

# Bước 4: In model summary
model.summary()

# Bước 5: Lưu mô hình
model.save("model/emotion_model.h5")
