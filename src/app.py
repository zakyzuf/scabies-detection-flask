# from flask import Flask, request, jsonify
# from tensorflow.keras.models import load_model
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# from scipy.special import expit 
# from PIL import Image

# import tensorflow as tf
# import numpy as np

# app = Flask(__name__)

# model = load_model("../models/model_scabies_inference.keras")

# labels = ["NON_SCABIES", "SCABIES"]

# @app.route("/")
# def main():
#     return "You have reached main page !"

# @app.route('/predict-first', methods=['POST'])
# def prediction():
#     if 'image' not in request.files:
#         return 'No image uploaded', 400

#     file = request.files['image']

#     # Load gambar menggunakan PIL
#     gambar = Image.open(file)
#     gambar = gambar.convert("RGB")
#     # Resize gambar jika diperlukan
#     ukuran_baru = (160, 160)  # ukuran yang diinginkan
#     gambar = gambar.resize(ukuran_baru)

#     # Konversi gambar ke array numpy
#     gambar_array = np.array(gambar)

#     # Tambahkan dimensi batch
#     gambar_array = np.expand_dims(gambar_array, axis=0)

#     # Preprocess gambar untuk MobileNetV2
#     gambar_array = preprocess_input(gambar_array)

#     # Lakukan prediksi menggunakan model machine learning
#     predictions = model.predict_on_batch(gambar_array).flatten()

#     # Apply a sigmoid since our model returns logits
#     probs = expit(predictions)
#     #predictions = tf.nn.sigmoid(predictions)
#     predictions = (probs >= 0.071).astype(int)

#     predicted_label = labels[predictions[0]]

#     return jsonify({
#         'predicted_label': predicted_label,
#         'probability': float(probs[0])  # Convert to float untuk memastikan kompatibilitas JSON
#     }), 200

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'image' not in request.files:
#         return 'No image uploaded', 400

#     file = request.files['image']

#     gambar = Image.open(file)

#     ukuran_baru = (160, 160)
#     gambar = gambar.resize(ukuran_baru)

#     gambar_array = np.array(gambar)

#     gambar_array = np.expand_dims(gambar_array, axis=0)

#     predictions = model.predict(gambar_array)

#     if predictions > 0:
#         predicted_label = 'SCABIES'
#     elif predictions == 0:
#         predicted_label = 'NON-SCABIES'

#     return jsonify({'predicted_label': predicted_label}), 200

# if __name__ == '__main__':
#     app.run(debug=True, port=5006)

from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
import cv2
from PIL import Image
import io

app = Flask(__name__)

# Load model saat aplikasi start
model = load_model('../models/scabies_model.h5')

def preprocess_image(image):
    # Resize ke 224x224 seperti saat training
    image = cv2.resize(image, (224, 224))
    # Konversi ke array dan normalisasi
    image = np.array(image) / 255.0
    # Expand dimensi untuk batch
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/predict-first', methods=['POST'])
def predict():
    try:
        # Terima file gambar dari request
        file = request.files['image']
        # Baca gambar
        image = Image.open(io.BytesIO(file.read()))
        # Konversi ke RGB
        image = image.convert('RGB')
        # Konversi ke numpy array
        image = np.array(image)
        
        # Preprocess gambar
        processed_image = preprocess_image(image)
        
        # Prediksi
        prediction = model.predict(processed_image)
        
        # Ambil class dengan probabilitas tertinggi
        predicted_class = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_class])
        
        # Mapping class ke label
        class_mapping = {0: 'NON_SCABIES', 1: 'SCABIES'}
        predicted_label = class_mapping[predicted_class]
        
        return jsonify({
            'predicted_label': predicted_label,
            'confidence': confidence,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        })

if __name__ == '__main__':
    app.run(debug=True, port=5006)