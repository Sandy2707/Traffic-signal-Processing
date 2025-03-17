import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

data = []
labels = []
classes = 43
cur_path = os.getcwd()

# Retrieving the images and their labels
for i in range(classes):
    path = os.path.join(r"C:\Users\Kurub\Downloads\archive (1)", 'train', str(i))
    
    if not os.path.exists(path):  # Check if directory exists
        print(f"Warning: Directory {path} not found.")
        continue
    
    images = os.listdir(path)
    
    for a in images:
        img_path = os.path.join(path, a)
        try:
            image = Image.open(img_path)
            image = image.resize((30, 30))
            image = np.array(image)
            data.append(image)
            labels.append(i)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")

# Converting lists into numpy arrays
data = np.array(data)
labels = np.array(labels)

# Check if dataset is empty
if data.shape[0] == 0:
    raise ValueError("No images loaded! Check dataset paths and image files.")

print(data.shape, labels.shape)

# Splitting training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Convert labels to categorical
y_train = to_categorical(y_train, classes)
y_test = to_categorical(y_test, classes)

# Building the model
model = Sequential([
    Conv2D(32, (5,5), activation='relu', input_shape=X_train.shape[1:]),
    Conv2D(32, (5,5), activation='relu'),
    MaxPool2D(pool_size=(2,2)),
    Dropout(0.25),
    Conv2D(64, (3,3), activation='relu'),
    Conv2D(64, (3,3), activation='relu'),
    MaxPool2D(pool_size=(2,2)),
    Dropout(0.25),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(classes, activation='softmax')
])

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
epochs = 15
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))

# Save model
model.save("traffic_classifier.h5")

# Plot accuracy
plt.figure(0)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot loss
plt.figure(1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Load test data
y_test_data = pd.read_csv('Test.csv')
labels = y_test_data["ClassId"].values
imgs = y_test_data["Path"].values

X_test = []

for img in imgs:
    try:
        image = Image.open(img)
        image = image.resize((30,30))
        X_test.append(np.array(image))
    except Exception as e:
        print(f"Error loading test image {img}: {e}")

X_test = np.array(X_test)

# Predict
pred = model.predict(X_test)
pred_classes = np.argmax(pred, axis=1)

# Accuracy
from sklearn.metrics import accuracy_score
print("Test Accuracy:", accuracy_score(labels, pred_classes))
