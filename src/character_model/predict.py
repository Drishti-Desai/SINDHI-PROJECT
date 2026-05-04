import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

def load_class_indices(output_dir):
    """Loads the mapping from class indices to class names."""
    class_indices_path = os.path.join(output_dir, 'class_indices.json')
    if not os.path.exists(class_indices_path):
        print(f"Warning: {class_indices_path} not found.")
        # Fallback to basic 1-52 mapping if json doesn't exist
        return {i: str(i+1) for i in range(52)}
        
    with open(class_indices_path, 'r') as f:
        class_indices = json.load(f)
    # Invert the dictionary to map index to class name
    return {v: k for k, v in class_indices.items()}

def predict_image(img_path):
    """
    Predicts the class of a given handwritten character image.
    
    Args:
        img_path (str): Path to the image file.
        
    Returns:
        str: Predicted class label.
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    model_path = os.path.join(base_dir, 'outputs', 'model.h5')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
        
    # Load the trained model
    model = load_model(model_path)
    
    # Preprocessing target size used during training
    target_size = (64, 64)
    
    # Load and preprocess the image
    try:
        img = image.load_img(img_path, target_size=target_size)
    except Exception as e:
        raise ValueError(f"Could not load image at {img_path}. Error: {e}")
        
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    img_array /= 255.0  # Rescale matching the ImageDataGenerator
    
    # Predict the class probabilities
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    
    # Map index back to class name
    output_dir = os.path.join(base_dir, 'outputs')
    idx_to_class = load_class_indices(output_dir)
    predicted_class = idx_to_class.get(predicted_class_index, str(predicted_class_index))
    
    print(f"Image: {img_path}")
    print(f"Predicted Class: {predicted_class}")
    
    return predicted_class

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Predict the class of a Sindhi handwritten character.')
    parser.add_argument('image_path', type=str, help='Path to the image to classify.')
    args = parser.parse_args()
    
    predict_image(args.image_path)
