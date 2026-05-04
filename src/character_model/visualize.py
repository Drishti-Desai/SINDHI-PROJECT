import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

def visualize_samples(data_dir, num_classes=10):
    """
    Visualizes 1 sample image per class for the specified number of classes.
    """
    print(f"Visualizing samples from: {data_dir}")
    
    # Get the list of class folders
    classes = sorted(os.listdir(data_dir))
    
    # Filter out hidden files like .DS_Store
    classes = [c for c in classes if not c.startswith('.') and os.path.isdir(os.path.join(data_dir, c))]
    
    # Take first num_classes
    classes = classes[:num_classes]
    
    plt.figure(figsize=(15, 6))
    
    for i, cls in enumerate(classes):
        cls_dir = os.path.join(data_dir, cls)
        # Get first image
        images = [img for img in os.listdir(cls_dir) if not img.startswith('.')]
        if not images:
            continue
            
        img_path = os.path.join(cls_dir, images[0])
        
        try:
            img = image.load_img(img_path)
            plt.subplot(2, 5, i + 1)
            plt.imshow(img, cmap='gray')
            plt.title(f"Class: {cls}")
            plt.axis('off')
        except Exception as e:
            print(f"Could not load image {img_path}: {e}")
            
    # Save the visualization to outputs folder
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    output_dir = os.path.join(base_dir, 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    save_path = os.path.join(output_dir, 'dataset_samples.png')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved visualization to: {save_path}")
    plt.show()

if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    train_dir = os.path.join(base_dir, 'data', 'alphabets', 'train')
    
    if os.path.exists(train_dir):
        visualize_samples(train_dir, num_classes=10)
    else:
        print(f"Train directory not found: {train_dir}")
