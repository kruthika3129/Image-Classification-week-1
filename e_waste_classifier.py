
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import gradio as gr
from PIL import Image
import os
import zipfile
import shutil

# =========================================
# NEW SECTION: KAGGLE SETUP & DATA DOWNLOAD
# =========================================
print("Setting up Kaggle and downloading dataset...")

# 1. Setup Kaggle API credentials
# Assumes 'kaggle.json' is uploaded to the current working directory
if os.path.exists('kaggle.json'):
    kaggle_dir = os.path.expanduser('~/.kaggle')
    os.makedirs(kaggle_dir, exist_ok=True)
    shutil.copy('kaggle.json', os.path.join(kaggle_dir, 'kaggle.json'))
    # Set permissions to strict for Kaggle API to work
    try:
        os.chmod(os.path.join(kaggle_dir, 'kaggle.json'), 0o600)
    except Exception as e:
        print(f"Note: Could not change permissions (might be on Windows), ignoring: {e}")
else:
    print("WARNING: 'kaggle.json' not found in current directory. Ensure API is set up manually if download fails.")

# 2. Download Dataset
# Using os.system for broad compatibility (works in Colab and standard terminals with Kaggle installed)
if not os.path.exists('e-waste-image-dataset.zip'):
    print("Downloading dataset...")
    exit_code = os.system('kaggle datasets download -d akshat103/e-waste-image-dataset')
    if exit_code != 0:
        raise Exception("Kaggle download failed. Check your kaggle.json and internet connection.")
else:
    print("Dataset zip already exists.")

# 3. Unzip Dataset
EXTRACT_DIR = os.path.join(os.getcwd(), 'e_waste_data')
if not os.path.exists(EXTRACT_DIR):
    print(f"Extracting to {EXTRACT_DIR}...")
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    with zipfile.ZipFile('e-waste-image-dataset.zip', 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
    print("Extraction complete.")
else:
    print(f"Data already extracted at {EXTRACT_DIR}")

# =========================================
# UPDATED SECTION: DATASET PATHS
# =========================================
# The dataset extracts into a folder named 'modified-dataset' usually.
# We dynamically find it to be safe.
base_path = EXTRACT_DIR
if 'modified-dataset' in os.listdir(base_path):
     base_path = os.path.join(base_path, 'modified-dataset')

trainpath = os.path.join(base_path, 'train')
validpath = os.path.join(base_path, 'val')
testpath = os.path.join(base_path, 'test')

print(f"Training path set to: {trainpath}")
print(f"Validation path set to: {validpath}")
print(f"Test path set to: {testpath}")

# Verify paths exist
if not os.path.exists(trainpath):
    raise FileNotFoundError(f"Could not find train folder at {trainpath}. Check extraction.")

# =========================================
# ORIGINAL SCRIPT CONTINUES BELOW
# =========================================

# Load datasets
datatrain= tf.keras.utils.image_dataset_from_directory(trainpath, shuffle=True, image_size=(128,128), batch_size=32)
datatest= tf.keras.utils.image_dataset_from_directory(testpath, shuffle=False, image_size=(128,128), batch_size=32)
datavalid = tf.keras.utils.image_dataset_from_directory(validpath, shuffle=True, image_size=(128,128), batch_size=32)

class_names = datatrain.class_names

# Visualize images
plt.figure(figsize=(10, 10))
for images, labels in datatrain.take(1):
    for i in range(12):
        ax = plt.subplot(4, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()

# Class distribution
def plot_class_distribution(dataset, title="Class Distribution"):
    class_counts = {}
    for images, labels in dataset:
        for label in labels.numpy():
            class_name = dataset.class_names[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    plt.figure(figsize=(10, 6))
    plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
    plt.xlabel("Class")
    plt.ylabel("Number of Items")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

plot_class_distribution(datatrain, "Training Data Distribution")
plot_class_distribution(datavalid, "Validation Data Distribution")
plot_class_distribution(datatest, "Test Data Distribution")

# Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Model setup
base_model = EfficientNetV2B0(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

model = Sequential([
    layers.Input(shape=(128, 128, 3)),
    data_augmentation,
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

early = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(datatrain, validation_data=datavalid, epochs=15, batch_size=100, callbacks=[early])

# Accuracy & Loss plots
acc = history.history.get('accuracy')
val_acc = history.history.get('val_accuracy')
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training vs Validation Accuracy')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training vs Validation Loss')
plt.show()

# Evaluation
loss, accuracy = model.evaluate(datatest)
print(f'Test accuracy is {accuracy:.4f}, Test loss is {loss:.4f}')

# Confusion matrix
y_true = np.concatenate([y.numpy() for x, y in datatest], axis=0)
y_pred_probs = model.predict(datatest)
y_pred = np.argmax(y_pred_probs, axis=1)

print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Sample predictions
for images, labels in datatest.take(1):
    predictions = model.predict(images)
    pred_labels = tf.argmax(predictions, axis=1)
    for i in range(8):
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(f"True: {class_names[labels[i]]}, Pred: {class_names[pred_labels[i]]}")
        plt.axis("off")
        plt.show()

model.save('Efficient_classify.keras')

