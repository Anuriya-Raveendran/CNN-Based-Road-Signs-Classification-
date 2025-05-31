Road Sign Classification using CNN

This project presents a deep learning-based solution to classify road signs using Convolutional Neural Networks (CNN). The system is designed to automatically recognize different road signs, which is a critical task for autonomous vehicles and advanced driver-assistance systems.

📌 Overview

This project involves building and training a CNN model to identify and classify traffic signs using a labeled dataset. The model is fine-tuned using hyperparameter optimization and evaluated using standard metrics to ensure high performance.

🧠 Technologies Used

Python

TensorFlow / Keras

Numpy, Pandas

Matplotlib, Seaborn

Scikit-learn

Keras Tuner (for Hyperparameter Optimization)

Streamlit (for User Interface)

🗂️ Dataset

The dataset used includes a large number of labeled traffic sign images. The images are categorized into multiple classes, each representing a specific road sign such as Stop, Speed Limit, Yield, etc.

🧱 Model Architecture

A custom CNN architecture was built using the following layers:

Convolutional Layers (Conv2D)

Max Pooling Layers (MaxPooling2D)

Dropout Layers (to prevent overfitting)

Fully Connected Layers (Dense)

Training Strategy:

Optimizer: Adam

Loss Function: Categorical Cross-Entropy

Epochs: 20

Batch Size: 32

Early Stopping applied to avoid overfitting

Hyperparameter Tuning:

Technique: Random Search with Keras Tuner

Parameters Tuned: Filters, Kernel Size, Dropout Rate, Learning Rate, Dense Units

📊 Evaluation Metrics

Accuracy (Train, Validation, and Test)

Loss Curves

Confusion Matrix with class names

🔍 Results

Training Accuracy: ~99.8%

Validation Accuracy: ~92.7%

Test Accuracy: ~94.6%

These results indicate a well-generalized model that performs accurately on unseen data.

📈 Performance Visualization

The model's learning progress is tracked using accuracy and loss curves, plotted using matplotlib. These plots help in understanding the learning behavior and identifying any signs of overfitting.

🧪 Confusion Matrix

A confusion matrix is generated with actual label names instead of integers, which provides a clear insight into the model's classification performance across different sign categories.

🌐 Streamlit Web App

A simple and interactive web UI was built using Streamlit to:

Upload test images

Display model predictions

Show confidence scores and performance metrics

🚀 Future Enhancements

Use of pre-trained models like VGG16, ResNet for improved accuracy using transfer learning.

Deploy the Streamlit app on cloud platforms.

Expand the dataset for better generalization.

📎 Conclusion

This CNN-based Road Sign Classifier demonstrates the practical application of deep learning in intelligent transportation systems. By achieving high accuracy through optimized training, this project lays the groundwork for integration into real-time systems like self-driving cars and road safety monitoring.




