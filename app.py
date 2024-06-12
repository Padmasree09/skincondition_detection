# from flask import Flask, render_template, request
# import tensorflow as tf
# from tensorflow import keras
# from keras.models import load_model
# from keras.preprocessing import image
# import numpy as np
# import os


# app = Flask(__name__)


# # Load the trained model
# model_path = 'models/my_model.h5'  # Adjust the path if necessary
# if os.path.exists(model_path):
#     model = load_model(model_path)
# else:
#     raise FileNotFoundError(f"Model file not found at {model_path}")


# # Define a function to preprocess the image
# def preprocess_image(img_path):
#     img = image.load_img(img_path, target_size=(150, 150))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.0
#     return img_array

# @app.route('/')
# def home():
#     return render_template('index.html')


# @app.route('/predict',methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return redirect(request.url)
    
#     file = request.files['file']
#     if file.filename == '':
#         return redirect(request.url)
    
#     if file:
#         file_path = os.path.join('uploads', file.filename)
#         file.save(file_path)
        
#         # Preprocess the image
#         img_array = preprocess_image(file_path)
        
#         # Make a prediction
#         prediction = model.predict(img_array)
#         predicted_class = np.argmax(prediction, axis=1)[0]
        
#         # Map predicted class index to class name (adjust based on your class indices)
#         class_indices = {0: 'Class 0', 1: 'Class 1', 2: 'Class 2'}  # Replace with your actual class names
#         result = class_indices[predicted_class]
        
#         return render_template('index.html', prediction=result)

# if __name__ == "__main__":
#     if not os.path.exists('uploads'):
#         os.makedirs('uploads')
#     app.run(host='0.0.0.0', port=8080, debug=True)

from flask import Flask, render_template, request, redirect
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model_path = 'models/my_model.h5'  # Ensure this path is correct
try:
    model = load_model(model_path)
    model.save('models/my_model_updated.h5')
except Exception as e:
    raise FileNotFoundError(f"Failed to load model at {model_path}. Error: {str(e)}")

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        
        img_array = preprocess_image(file_path)
        
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        
        class_indices = {0: 'Acne', 1: 'Clear', 2: 'Comedone'}
        result = class_indices.get(predicted_class, 'Unknown condition')
        
        return render_template('index.html', prediction=result)

if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(host='0.0.0.0', port=8080, debug=True)
