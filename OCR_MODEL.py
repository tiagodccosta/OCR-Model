from emnist import extract_training_samples, extract_test_samples
import tensorflow as tf
from tensorflow.keras.utils import to_categorical  # type: ignore
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Load and preprocess the EMNIST dataset
train_images, train_labels = extract_training_samples('byclass')
test_images, test_labels = extract_test_samples('byclass')

# Normalize pixel values (0-1 range)
train_images = train_images / 255.0
test_images = test_images / 255.0

# One-hot encode labels
num_classes = 62  # 10 digits + 26 uppercase letters + 26 lowercase letters
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)

# Reshape images to match TensorFlow expectations
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

# Load the previously saved model
model = tf.keras.models.load_model('ocr_model.keras')
print("Model loaded successfully!")

# Compile the model (necessary before resuming training)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model for 10 more epochs
history = model.fit(
    train_images, train_labels,
    epochs=1,  # Training for 10 more epochs
    batch_size=128,
    validation_data=(test_images, test_labels)
)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy after additional 10 epochs: {test_acc * 100:.2f}%")

# Visualize training vs validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Visualize predictions on test data
predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(test_labels, axis=1)

# Create a dictionary to map indices to corresponding letters
index_to_label = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}
index_to_label.update({i + 10: chr(65 + i) for i in range(26)})  # Map indices 10-35 to A-Z
index_to_label.update({i + 36: chr(97 + (i - 26)) for i in range(26)})  # Map indices 36-61 to a-z

# Map predicted indices and true indices to letters
predicted_labels = [index_to_label[i] for i in predicted_classes]
true_labels = [index_to_label[i] for i in true_classes]

# Plot a few test samples with predictions
num_samples = 20
cols = 5
rows = num_samples // cols + (num_samples % cols > 0)  # Calculate the number of rows needed
plt.figure(figsize=(15, 3 * rows))  # Adjust the figure size dynamically

for i in range(num_samples):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(test_images[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
    true_label = true_labels[i]
    pred_label = predicted_labels[i]
    color = 'green' if true_label == pred_label else 'red'
    plt.title(f"True: {true_label}\nPred: {pred_label}", color=color)

plt.tight_layout()
plt.show()

# Save the fine-tuned model
model.save('ocr_model_finetuned.keras')
print("Fine-tuned model saved as 'ocr_model_finetuned.keras'")
