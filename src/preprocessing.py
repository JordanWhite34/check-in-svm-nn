import os
import cv2
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator

# Define paths
base_dir = '../data/raw'
clean_dir = os.path.join(base_dir, 'clean')
messy_dir = os.path.join(base_dir, 'messy')
processed_base_dir = '../data/preprocessed'
processed_clean_dir = os.path.join(processed_base_dir, 'clean')
processed_messy_dir = os.path.join(processed_base_dir, 'messy')

# Create directories for processed images if they don't exist
os.makedirs(processed_clean_dir, exist_ok=True)
os.makedirs(processed_messy_dir, exist_ok=True)


# Image preprocessing function
def preprocess_image(image_path, output_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224)) # Resize image to 224x224
    image = image / 255.0 # Normalize pixel values
    cv2.imwrite(output_path, image * 255) # Save preprocessed image


# Data augmentation generator
data_gen = ImageDataGenerator(rotation_range=20,
                              width_shift_range=0.2,
                              height_shift_range=0.2,
                              shear_range=0.2,
                              zoom_range=0.2,
                              horizontal_flip=True,
                              fill_mode='nearest')


def augment_and_save(directory, class_dir, processed_class_dir):
    for filename in os.listdir(class_dir):
        image_path = os.path.join(class_dir, filename)
        output_path = os.path.join(processed_class_dir, filename)
        image = np.expand_dims(cv2.imread(image_path), axis=0)
        save_prefix = os.path.splitext(filename)[0]
        i = 0
        for batch in data_gen.flow(image, batch_size=1,
                                   save_to_dir=processed_class_dir,
                                   save_prefix=save_prefix,
                                   save_format='jpeg'):
            i += 1
            if i > 5: # Generate 5 augmented images per original image
                break
        preprocess_image(image_path, output_path) # Also save the original preprocessed image


# Process and augment images for each category
augment_and_save('clean', clean_dir, processed_clean_dir)
augment_and_save('messy', messy_dir, processed_messy_dir)

print("Preprocessing and data augmentation completed.")
