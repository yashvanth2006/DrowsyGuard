import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# ── Config ────────────────────────────────────────────────
TRAIN_PATH = "archive/data/train"
VAL_PATH   = "archive/data/val"
IMG_SIZE   = 24
EPOCHS     = 15
BATCH_SIZE = 32

# ── Load images from awake/sleepy folders ─────────────────
def load_data(path):
    data, labels = [], []
    for label, folder in enumerate(["awake", "sleepy"]):
        folder_path = os.path.join(path, folder)
        if not os.path.exists(folder_path):
            print(f"❌ Folder not found: {folder_path}")
            continue
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            data.append(img)
            labels.append(label)   # 0=awake, 1=sleepy
    return np.array(data), np.array(labels)

print("📂 Loading training data...")
X_train, y_train = load_data(TRAIN_PATH)
print(f"   Awake : {list(y_train).count(0)}")
print(f"   Sleepy: {list(y_train).count(1)}")

print("📂 Loading validation data...")
X_val, y_val = load_data(VAL_PATH)

# ── Preprocess ────────────────────────────────────────────
X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype("float32") / 255.0
X_val   = X_val.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype("float32") / 255.0
y_train = to_categorical(y_train, 2)
y_val   = to_categorical(y_val,   2)

print(f"\n📊 Train: {len(X_train)} | Val: {len(X_val)}")

# ── CNN Model ─────────────────────────────────────────────
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ── Train ─────────────────────────────────────────────────
print("\n🚀 Training started...")
history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, y_val)
)

# ── Evaluate ──────────────────────────────────────────────
loss, acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\n✅ Validation Accuracy: {acc*100:.2f}%")
print(f"   Validation Loss:     {loss:.4f}")

# ── Save model ────────────────────────────────────────────
model.save("eye_state_model.h5")
print("💾 Model saved as eye_state_model.h5")

# ── Plot training curves (for your report) ────────────────
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'],     label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'],     label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("training_curves.png")
print("📈 Training curves saved as training_curves.png")
plt.show()