from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('model_logreg.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.is_json:
        data = request.get_json()
    else:
        data = request.form.to_dict()

    # Ambil nilai fitur sesuai urutan UCI Credit Card dataset
    features = [
        float(data.get('LIMIT_BAL', 0)),
        float(data.get('SEX', 0)),
        float(data.get('EDUCATION', 0)),
        float(data.get('MARRIAGE', 0)),
        float(data.get('AGE', 0)),
        float(data.get('PAY_0', 0)),
        float(data.get('PAY_2', 0)),
        float(data.get('PAY_3', 0)),
        float(data.get('PAY_4', 0)),
        float(data.get('PAY_5', 0)),
        float(data.get('PAY_6', 0)),
        float(data.get('BILL_AMT1', 0)),
        float(data.get('BILL_AMT2', 0)),
        float(data.get('BILL_AMT3', 0)),
        float(data.get('BILL_AMT4', 0)),
        float(data.get('BILL_AMT5', 0)),
        float(data.get('BILL_AMT6', 0)),
        float(data.get('PAY_AMT1', 0)),
        float(data.get('PAY_AMT2', 0)),
        float(data.get('PAY_AMT3', 0)),
        float(data.get('PAY_AMT4', 0)),
        float(data.get('PAY_AMT5', 0)),
        float(data.get('PAY_AMT6', 0))
    ]

    X = scaler.transform([features])
    proba = model.predict_proba(X)[0][1]
    pred = int(model.predict(X)[0])

    label = "Berisiko Gagal Bayar" if pred == 1 else "Tidak Berisiko"
    return jsonify({
        "prediction": pred,
        "probability": float(proba),
        "label": label
    })

if __name__ == '__main__':
    app.run(debug=True)
