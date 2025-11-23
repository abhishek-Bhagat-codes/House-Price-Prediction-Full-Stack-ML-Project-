from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

app = Flask(__name__)
app = Flask(__name__, template_folder='views')
model = pickle.load(open("model/house_price_model.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")   # 👈 LOAD HTML PAGE


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Convert JSON to array
    features = np.array([[
        data['longitude'],
        data['latitude'],
        data['housing_median_age'],
        data['total_rooms'],
        data['total_bedrooms'],
        data['population'],
        data['households'],
        data['median_income'],
        data['ocean_proximity']
    ]])

    prediction = model.predict(features)[0]

    return jsonify({'predicted_price': float(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
