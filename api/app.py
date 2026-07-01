from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load trained model
model_path = os.path.join("..", "models", "churn_model.pkl")
model = joblib.load(model_path)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = {
        "gender": [request.form["gender"]],
        "SeniorCitizen": [int(request.form["SeniorCitizen"])],
        "Partner": [request.form["Partner"]],
        "Dependents": [request.form["Dependents"]],
        "tenure": [int(request.form["tenure"])],
        "PhoneService": [request.form["PhoneService"]],
        "MultipleLines": [request.form["MultipleLines"]],
        "InternetService": [request.form["InternetService"]],
        "OnlineSecurity": [request.form["OnlineSecurity"]],
        "OnlineBackup": [request.form["OnlineBackup"]],
        "DeviceProtection": [request.form["DeviceProtection"]],
        "TechSupport": [request.form["TechSupport"]],
        "StreamingTV": [request.form["StreamingTV"]],
        "StreamingMovies": [request.form["StreamingMovies"]],
        "Contract": [request.form["Contract"]],
        "PaperlessBilling": [request.form["PaperlessBilling"]],
        "PaymentMethod": [request.form["PaymentMethod"]],
        "MonthlyCharges": [float(request.form["MonthlyCharges"])],
        "TotalCharges": [float(request.form["TotalCharges"])]
    }

    df = pd.DataFrame(data)

    prediction = model.predict(df)[0]

    if str(prediction).strip().lower() in ["yes", "1", "true"]:
        result = "Customer Will Churn"
    else:
        result = "Customer Will Not Churn"

    return render_template("index.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)