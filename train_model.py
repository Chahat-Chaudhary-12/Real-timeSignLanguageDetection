import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models
import pickle

DATA_PATH = "landmark_data"

X = []
y = []
labels = []

label_map = {}

# IMPORTANT FIX: sorted()
for idx, word in enumerate(sorted(os.listdir(DATA_PATH))):
    label_map[idx] = word
    labels.append(word)

    word_path = os.path.join(DATA_PATH, word)

    for file in os.listdir(word_path):
        data = np.load(os.path.join(word_path, file))
        X.append(data)
        y.append(idx)

X = np.array(X)
y = np.array(y)
  
print("Input shape:", X.shape)

# MODEL (IMPROVED)
model = models.Sequential([
    layers.Dense(256, activation='relu', input_shape=(42,)),
    layers.Dropout(0.4),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(labels), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X, y, epochs=50, batch_size=16, validation_split=0.2)

model.save("landmark_model.keras")

with open("labels.pkl", "wb") as f:
    pickle.dump(label_map, f)

print("Training Done ")