import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import build_cnn_model

def plot_history(history, save_dir):
    """
    Plots training and validation accuracy and loss, and saves the graphs.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(1, len(acc) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Save the plots
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.show()

def train_model():
    """
    Main function to prepare data, build the model, and train it.
    """
    # Define paths relative to this script
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_dir = os.path.join(base_dir, 'data', 'alphabets')
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')
    output_dir = os.path.join(base_dir, 'outputs')
    
    # Ensure outputs directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Hyperparameters
    batch_size = 32
    target_size = (64, 64)
    epochs = 50  # Increased epochs for better training
    num_classes = 52
    
    print("Setting up data generators with augmentation...")
    # Initialize ImageDataGenerators
    # Added augmentation for the training set
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False, # Handwritten characters are usually not flipped
        fill_mode='nearest'
    )
    
    # Validation and test data only need rescaling
    valid_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load data from directories
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    valid_generator = valid_datagen.flow_from_directory(
        valid_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    print("Building model...")
    model = build_cnn_model(input_shape=(target_size[0], target_size[1], 3), num_classes=num_classes)
    model.summary()
    
    # Define Callbacks
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    
    print("Starting training...")
    # Train the model
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=valid_generator,
        callbacks=[early_stop, reduce_lr]
    )
    
    # Save the training graphs
    print("Saving training graphs...")
    plot_history(history, output_dir)
    
    # Evaluate on the test dataset
    print("Evaluating model on test dataset...")
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Save the trained model
    model_save_path = os.path.join(output_dir, 'model.h5')
    model.save(model_save_path)
    print(f"Model saved successfully at: {model_save_path}")
    
    # Save the class indices mapping
    import json
    class_indices_path = os.path.join(output_dir, 'class_indices.json')
    with open(class_indices_path, 'w') as f:
        json.dump(train_generator.class_indices, f)
    print(f"Class indices saved at: {class_indices_path}")

if __name__ == '__main__':
    train_model()
