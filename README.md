SINDHIMP is a Deep Learning–based OCR (Optical Character Recognition) system developed to recognize handwritten Sindhi characters and convert them into editable digital text.

The project combines Computer Vision, Deep Learning, and OCR technologies to help preserve and digitize historical Sindhi manuscripts and handwritten documents.

SINDHIMP is a beginner-level research and development project focused on handwritten Sindhi character recognition using Deep Learning and Optical Character Recognition (OCR).

The project serves as a foundational prototype for future large-scale digitization of handwritten Sindhi manuscripts and documents. It is intended to be further enhanced and expanded for potential use in collaboration with the Institute of Sindhology for preservation of Sindhi linguistic and cultural heritage.

The system uses:
OpenCV for image preprocessing
CNN (Convolutional Neural Network) for character recognition
Tesseract OCR for extracting Sindhi text
Streamlit for the user interface

 
Objectives : 
Recognize handwritten Sindhi characters
Classify 52 distinct Sindhi character classes
Convert handwritten text into editable digital text
Build a simple and interactive web interface
Create a scalable OCR pipeline for future manuscript digitization


Features : 
Handwritten Sindhi character recognition
OCR-based text extraction
Image preprocessing using OpenCV
CNN-based deep learning model
Streamlit web interface
Hybrid local + cloud training approach
Supports noisy and cursive handwritten inputs

System Workflow : 
User uploads handwritten Sindhi image
Image preprocessing using OpenCV:
Grayscale conversion
Noise removal
Otsu Thresholding
Image resizing (64×64)
CNN extracts image features
Character classification performed
Tesseract OCR extracts text
Final digital text displayed to user
