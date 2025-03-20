# Deepfake Detection System

## Overview
This project implements a deep learning system for detecting deepfake images using the Xception architecture with transfer learning. The model is trained to classify images as either real or artificially generated (deepfakes). With the increasing sophistication of deepfake technology, this system provides a reliable method to distinguish between authentic and manipulated images, helping combat the spread of misinformation.

## Dataset
The system is trained on a dataset containing:
- 140,002 training images
- 10,905 test images
- 39,428 validation images

The dataset is organized with two classes: real and deepfake images. The deepfake images include various manipulation techniques such as face swapping, attribute manipulation, and GAN-generated faces. All images are preprocessed to 299x299 pixels to match the Xception model's input requirements.

The dataset used is a combination of:
- Real and fake celebrity images from the Kaggle Deepfake Detection Challenge dataset
- Images from the DFDC (DeepFake Detection Challenge) dataset
- Additional synthetic images generated using state-of-the-art GAN methods

## Model Architecture
- Base model: Xception (pre-trained on ImageNet)
  - The Xception architecture utilizes depthwise separable convolutions which provide excellent performance while maintaining computational efficiency
  - 71 layers deep with approximately 22.9M trainable parameters
- Custom top layers for classification:
  - Global Average Pooling to reduce spatial dimensions
  - Dense layer (32 neurons) with ReLU activation
  - Dense layer (16 neurons) with ReLU activation
  - Output layer with sigmoid activation for binary classification (0 = real, 1 = fake)
- Training strategy:
  - Initial freezing of base model layers to preserve pre-trained features
  - Fine-tuning of the last 20 layers to adapt to the specific task of deepfake detection
  - Batch size of 32 for optimal training efficiency

## Features
- Transfer learning using Xception architecture pre-trained on ImageNet
- Fine-tuning of the last 20 layers of the base model
- Data augmentation for training to enhance model generalization
- Binary classification (real vs fake) with confidence scores
- Evaluation metrics including:
  - ROC curve and AUC score
  - Precision-recall curve
  - Confusion matrix
  - Classification report with precision, recall, and F1-score
- Visualization of model performance and training history
- Attention heatmaps to visualize areas the model focuses on for detection

## Performance
The model achieves:
- 97.8% accuracy on the test set
- 0.982 AUC (Area Under the ROC Curve)
- 0.973 precision and 0.984 recall for deepfake detection
- Fast inference time (~150ms per image on GPU)

## Requirements
- Python 3.8+
- TensorFlow 2.6+
- Keras 2.6+
- NumPy 1.19+
- Pandas 1.3+
- Matplotlib 3.4+
- Scikit-learn 0.24+
- OpenCV 4.5+
- CUDA 11.2+ (for GPU acceleration)
- 16GB RAM minimum (32GB recommended)

## Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/Deepfake-Detection-System.git
cd Deepfake-Detection-System

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
The entire workflow is documented in the `Deepfake-Detection-System.ipynb` notebook, which includes:
1. Data loading and preprocessing
   - Loading images from the dataset
   - Resizing to 299x299 pixels
   - Normalizing pixel values
   - Creating training, validation, and test datasets
2. Model architecture definition
   - Loading the pre-trained Xception model
   - Adding custom classification layers
   - Freezing base layers and configuring for fine-tuning
3. Model training
   - Setting up callbacks for checkpointing and early stopping
   - Training with data augmentation
   - Monitoring training and validation metrics
4. Evaluation and visualization of results
   - Generating confusion matrix
   - Plotting ROC and precision-recall curves
   - Creating classification reports

### For inference on new images:
```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model('deepfake_detection_model.h5')

# Preprocess the image
img = image.load_img('path_to_image.jpg', target_size=(299, 299))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0  # Normalize

# Make prediction
prediction = model.predict(img_array)
confidence = prediction[0][0]
result = "Fake" if confidence > 0.5 else "Real"
print(f"Image classified as: {result} with {confidence*100:.2f}% confidence")
```

## Limitations
- The model is specifically trained on faces and may not perform well on other types of deepfakes
- Performance may vary with different types of deepfake generation techniques
- Very high-quality deepfakes created with the latest technologies might still be challenging to detect
- Images with poor lighting or unusual angles might affect detection accuracy

## Future Work
- Incorporate video-based deepfake detection capabilities
- Implement explainable AI techniques to better understand detection decisions
- Add support for detecting text and audio deepfakes
- Develop ensemble methods combining multiple detection approaches
- Create a lightweight model version for mobile deployment

## Contributing
Contributions to improve the deepfake detection system are welcome:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements
- The Xception model architecture was developed by François Chollet
- Dataset partially sourced from the DeepFake Detection Challenge (DFDC) by Facebook/AWS/Microsoft/Media Integrity Steering Committee
- Special thanks to Kaggle for hosting the deepfake detection competitions
- This research was inspired by the paper "MesoNet: a Compact Facial Video Forgery Detection Network" by Afchar et al.
- The implementation references "FaceForensics++: Learning to Detect Manipulated Facial Images" by Rössler et al.

## References
1. Chollet, F. (2017). Xception: Deep learning with depthwise separable convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1251-1258).
2. Rössler, A., Cozzolino, D., Verdoliva, L., Riess, C., Thies, J., & Nießner, M. (2019). FaceForensics++: Learning to detect manipulated facial images. In Proceedings of the IEEE International Conference on Computer Vision (pp. 1-11).
3. Afchar, D., Nozick, V., Yamagishi, J., & Echizen, I. (2018). MesoNet: a Compact Facial Video Forgery Detection Network. In 2018 IEEE International Workshop on Information Forensics and Security (WIFS) (pp. 1-7).
