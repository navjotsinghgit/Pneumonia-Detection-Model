import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing import image
import io
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)
# Load the trained model
model = tf.keras.models.load_model(r'D:\HackAryaVerse\pneumonia_model.h5')

# Function to preprocess the image before prediction
def preprocess_image(img_bytes):
    img = image.load_img(io.BytesIO(img_bytes), target_size=(224, 224))  # Adjust size to your model's input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize if necessary (depending on your model)
    return img_array

def test(img_path):
    with open(img_path, 'rb') as f:
        img_bytes = f.read()
    img_array = preprocess_image(img_bytes)
    
    # Make a prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=-1)
    print("\n\n\n\n\n\n", str(predicted_class))
#test(r'D:\HackAryaVerse\test.jpg')
# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')  # Create an HTML page for the UI

# Route for model prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    img_bytes = file.read()
    img_array = preprocess_image(img_bytes)
    
    # Make a prediction
    prediction = model.predict(img_array)
    
    # Assuming a binary classification model for simplicity
    predicted_class = np.argmax(prediction, axis=-1)  # Or modify according to your output
    if (predicted_class<0.5):
        guess = "This person does has pnemonia"
    else:
        guess = "This Person Does not have pnemonia"
    # Return prediction result
    return jsonify({'prediction': guess})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
