from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

# Only using top 6 features
selected_features = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean smoothness",
    "mean compactness",
    "mean concave points"
]

@app.route('/')
def home():
    return render_template('index.html', features=selected_features)

@app.route('/predict', methods=['POST'])
def predict():
    values = [float(request.form[feature]) for feature in selected_features]
    data = np.array([values])
    prediction = model.predict(data)[0]
    result = "Benign Tumor ✅" if prediction == 1 else "Malignant Tumor ⚠️"
    return render_template('index.html', features=selected_features, result=result)

if __name__ == "__main__":
    app.run(debug=True)

