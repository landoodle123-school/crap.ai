from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Define the model
class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = layers.Conv2D(32, 3, activation='relu')
        self.flatten = layers.Flatten()
        self.d1 = layers.Dense(128, activation='relu')
        self.d2 = layers.Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

# Instantiate and build the model
model = MyModel()
model.build((None, 28, 28, 1))  # Adjust for the model input shape

# Load your trained model weights (check the file path)
try:
    model.load_weights('thing.keras.zip.keras')  # Update with correct path
except Exception as e:
    print(f"Error loading weights: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    
    try:
        img = Image.open(io.BytesIO(file.read())).convert('L')  # Convert to grayscale
        img = img.resize((28, 28))  # Resize to 28x28
        img_array = np.array(img) / 255.0  # Normalize
        img_array = img_array[..., np.newaxis]  # Add channel dimension
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make prediction
        predictions = model(img_array)  # Use the model instance directly
        predicted_class = np.argmax(predictions.numpy(), axis=-1)[0]  # Convert to numpy array

        return jsonify({'predicted_class': int(predicted_class)})

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {e}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
