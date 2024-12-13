import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.utils import to_categorical  # type: ignore
from emnist import extract_test_samples

# Load the pre-trained model
model_path = "ocr_model.keras"
model = load_model(model_path)

# Load the EMNIST dataset
test_images, test_labels = extract_test_samples('byclass')

# Preprocess the test data
test_images = test_images.astype('float32') / 255.0
test_images = np.expand_dims(test_images, -1)

# One-hot encode labels
num_classes = 62
test_labels_categorical = to_categorical(test_labels, num_classes)

# Make predictions on the test set
predictions = model.predict(test_images)

# Convert probabilities to class indices
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(test_labels_categorical, axis=1)

# Create a mapping from indices to characters
index_to_label = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}
index_to_label.update({i + 10: chr(65 + i) for i in range(26)})
index_to_label.update({i + 36: chr(97 + i) for i in range(26)})

# Number of samples to display
num_samples = 50

# Randomly select indices for the test images
random_indices = np.random.choice(len(test_images), size=num_samples, replace=False)

# Subset the data
sampled_images = test_images[random_indices]
sampled_true_classes = true_classes[random_indices]
sampled_pred_classes = predicted_classes[random_indices]

# Map predicted indices and true indices to letters
sampled_true_labels = [index_to_label[i] for i in sampled_true_classes]
sampled_pred_labels = [index_to_label[i] for i in sampled_pred_classes]

# Plot the selected samples with predictions
cols = 5
rows = num_samples // cols + (num_samples % cols > 0)
plt.figure(figsize=(15, 3 * rows))

for i in range(num_samples):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(sampled_images[i].squeeze(), cmap='gray')
    true_label = sampled_true_labels[i]
    pred_label = sampled_pred_labels[i]
    color = 'green' if true_label == pred_label else 'red'
    plt.title(f"True: {true_label}\nPred: {pred_label}", color=color)
    plt.axis('off')

plt.tight_layout()
plt.show()
