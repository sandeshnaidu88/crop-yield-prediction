from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = pickle.load(open("model.pkl", "rb"))
area_encoder = pickle.load(open("area_encoder.pkl", "rb"))
item_encoder = pickle.load(open("item_encoder.pkl", "rb"))
feature_columns = pickle.load(open("features.pkl", "rb"))

@app.route('/')
def home():
    return "Crop Yield Prediction API is Running"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        area = data['Area'].strip().lower()
        item = data['Item'].strip().lower()
        year = float(data['Year'])
        rainfall = float(data['Rainfall'])
        pesticide = float(data['Pesticide'])
        temp = float(data['Temperature'])

        if area not in area_encoder.classes_:
            return jsonify({"error": "Invalid Area entered"})

        if item not in item_encoder.classes_:
            return jsonify({"error": "Invalid Crop entered"})

        area_enc = area_encoder.transform([area])[0]
        item_enc = item_encoder.transform([item])[0]

        input_data = np.array([[area_enc, item_enc, year, rainfall, pesticide, temp]])

        prediction = model.predict(input_data)[0]

        return jsonify({
            "prediction": round(float(prediction), 2)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)