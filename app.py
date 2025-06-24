from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model, scaler, and column names
scaler = pickle.load(open("scaler.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html', features=columns)

# Manual form submission
@app.route('/predict', methods=['POST'])
def predict():
    values = [float(request.form[feature]) for feature in columns]
    data = np.array([values])
    data = scaler.transform(data)
    prediction = model.predict(data)[0]
    result = "Benign Tumor ✅" if prediction == 1 else "Malignant Tumor ⚠️"

    return render_template('result.html', features=columns, result=result, values=values)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        df = pd.read_csv(file)

        X = df[columns]
        data = scaler.transform(X)
        preds = model.predict(data)
        results = ['Benign ✅' if p == 1 else 'Malignant ⚠️' for p in preds]

        values = X.iloc[0].tolist()
        result = results[0]

        return render_template('result.html', features=columns, result=result, values=values)

    return render_template('upload.html')


# ✅ Always at bottom
if __name__ == "__main__":
    app.run(debug=True)
