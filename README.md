# Pneumonia-Detection-Model
Pneumonia detection model using CNN and CGAN. CNN trained for 25 epochs; CGAN for 500 epochs. Achieves 98% accuracy. Includes Gaussian filtering for noise removal, clean dataset filtering, RAG for user-driven data generation, and data augmentation for diversity and robustness.
Pneumonia Detection using CNN and CGAN
This project implements a deep learning-based system to detect pneumonia from chest X-ray images using a Convolutional Neural Network (CNN) and a Conditional Generative Adversarial Network (CGAN).

🚀 Features
CNN Classifier trained for 25 epochs with 98% accuracy.

CGAN trained over 500 epochs to generate synthetic pneumonia-infected images and highlight disease regions.

Gaussian Filter for advanced noise tracking.

Clean Dataset Function to remove low-quality or corrupted images before training.

Region-Adaptive Generation (RAG) to generate data from user input.

Data Augmentation to improve model generalization and diversity.

🧠 Model Architecture
CNN: Used for classification of pneumonia vs normal.

CGAN: Used for data generation and infected area visualization.

RAG: Enhances dataset with user-guided image region input.

📊 Performance
Accuracy: 98% on the validation set.

Supports visualization of infected areas with the CGAN module.

📁 Dataset
Cleaned and preprocessed X-ray images.

Augmented using techniques like rotation, flipping, and brightness/contrast adjustment.
