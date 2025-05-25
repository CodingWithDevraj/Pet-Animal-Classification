import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import os # Import the os module

# Load your trained model
model = load_model("cat_dog_snake_model.h5")

# Define class labels (make sure order matches your training)
class_names = ['cat', 'dog', 'snake']

from google.colab import files

# Assuming user uploads one image
# Update img_path to point to an actual image file
# You can pick one image from your dataset directory for testing
data_dir = '/content/dataset/dataset'
# Example: point to the first image found in the 'cat' folder
# Make sure the subdirectories (cat, dog, snake) exist within data_dir
try:
    # Get a list of subdirectories (class folders)
    class_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    if class_folders:
        # Pick the first class folder found
        first_class_folder = class_folders[2]
        class_folder_path = os.path.join(data_dir, first_class_folder)

        # Get a list of files in that folder
        image_files = [f for f in os.listdir(class_folder_path) if os.path.isfile(os.path.join(class_folder_path, f))]

        if image_files:
            # Pick the first image file found
            img_path = os.path.join(class_folder_path, image_files[100])
            print(f"Using image for prediction: {img_path}")
        else:
            print(f"Error: No image files found in {class_folder_path}")
            img_path = None # Set to None if no image is found
    else:
        print(f"Error: No class subdirectories found in {data_dir}")
        img_path = None # Set to None if no class folders are found
except Exception as e:
    print(f"An error occurred while trying to find an image file: {e}")
    img_path = None # Set to None if an error occurs

# Load image with OpenCV
# Only attempt to load if img_path is not None
if img_path:
    image = cv2.imread(img_path)
    if image is None:
        print(f"Error: Could not load image from {img_path}")
    else:
        image_resized = cv2.resize(image, (128, 128))  # Must match model input

        # Normalize and reshape
        img_array = image_resized / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        pred = model.predict(img_array)
        label = class_names[np.argmax(pred)]
        confidence = np.max(pred)

        # Draw label on image
        output = image.copy()
        cv2.putText(output, f"{label} ({confidence:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Convert BGR to RGB for display
        output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

        # Show image in notebook
        plt.imshow(output_rgb)
        plt.axis('off')
        plt.title("Prediction Result")
        plt.show()