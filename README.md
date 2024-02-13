
# Check-In Project

## Overview
This project implements a machine learning model to classify images of rooms as either clean or messy. It explores the use of Support Vector Machine (SVM) and Neural Networks (NN) for image classification, based on the study outlined in the provided SVM paper. The goal is to compare the performance of these algorithms in accurately categorizing room states using a custom dataset.

## Dataset
The dataset consists of labeled images categorized into 'Clean' and 'Messy'. Ensure your dataset is prepared with the following considerations:
- Images are resized to a uniform dimension.
- Normalization is applied to pixel values.
- (Optional) Grayscale conversion if color is not a significant feature.

## Dependencies
- Python 3.x
- Libraries: NumPy, TensorFlow/Keras, scikit-learn, OpenCV (for image processing)

## Implementation Steps
1. **Data Preprocessing**: Resize, normalize, and possibly convert images to grayscale.
2. **Feature Extraction**: Use VGG-16 for extracting features or directly feed images to a neural network.
3. **Model Training**:
   - SVM: Use scikit-learn for SVM implementation. Tune kernel and regularization parameters.
   - NN: Start with a simple architecture using TensorFlow or Keras. Adjust complexity as needed.
4. **Evaluation**: Use metrics like accuracy, precision, recall, and F1 score for evaluation.

## Usage
- To run the SVM model: `python svm_model.py`
- To run the Neural Network model: `python nn_model.py`

## Contribution
Feel free to contribute by extending the dataset, improving the models, or suggesting new features.

## License
Specify your project's license here.
