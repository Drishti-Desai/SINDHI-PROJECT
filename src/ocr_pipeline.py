import cv2
import pytesseract
import matplotlib.pyplot as plt
import os
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

# Set tesseract path for macOS Homebrew users if needed
if os.path.exists("/opt/homebrew/bin/tesseract"):
    pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

def preprocess_image(image_path):
    """
    Preprocesses the image for better OCR results.
    - Converts to grayscale
    - Applies thresholding
    - Denoising (to improve clarity)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")
        
    # Load the image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not decode image at: {image_path}")
        
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Thresholding (Otsu's thresholding)
    # This converts the image to binary (black and white)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Denoising to remove small dots/noise
    processed_img = cv2.medianBlur(thresh, 3)
    
    return processed_img, img

def extract_text(image_path, lang='snd'):
    """
    Extracts text from the image using pytesseract.
    Note: Requires the 'snd' (Sindhi) language data for tesseract.
    """
    processed_img, original_img = preprocess_image(image_path)
    
    # Perform OCR
    # config='--psm 6' assumes a single uniform block of text.
    # You can change lang to 'eng' for testing if 'snd' isn't installed yet.
    try:
        extracted_text = pytesseract.image_to_string(processed_img, lang=lang)
    except pytesseract.TesseractError:
        print(f"Warning: Language '{lang}' not found. Falling back to 'eng'.")
        extracted_text = pytesseract.image_to_string(processed_img, lang='eng')
    
    return extracted_text, processed_img, original_img

def main():
    # Path to the sample image
    image_path = "data/test_images/sample.jpg"
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Could not find image at {image_path}")
        print("Please place a Sindhi text image at that path (e.g., sample.jpg) and run again.")
        return

    print(f"--- Starting OCR Pipeline for: {image_path} ---")
    
    try:
        # Run OCR
        text, processed, original = extract_text(image_path, lang='snd')
        
        print("\n[Extracted Text]:")
        print("-" * 30)
        if text.strip():
            print(text.strip())
        else:
            print("(No text detected. Ensure the image is clear and the Sindhi language pack is installed.)")
        print("-" * 30)
        
        # Display the results
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(processed, cmap='gray')
        plt.title("Preprocessed (Binary)")
        plt.axis('off')
        
        plt.tight_layout()
        print("\nVisualizing results...")
        
        # Save visualization
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "ocr_result_visualization.png"))
        print(f"Visualization saved to {output_dir}/ocr_result_visualization.png")
        
        plt.show()
        
    except Exception as e:
        print(f"An error occurred: {e}")
        print("\nTip: Make sure tesseract and the Sindhi language pack ('snd') are installed.")
        print("On Mac: brew install tesseract-lang")

if __name__ == "__main__":
    main()
