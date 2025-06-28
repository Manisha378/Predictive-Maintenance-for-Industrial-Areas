from flask import Flask, render_template, request
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)
model = joblib.load("models/final_model.pkl")

@app.route("/")
def home():
    return render_template("login.html")

@app.route("/upload")
def upload():
    return render_template("upload.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/result", methods=["POST"])
def result():
    sensor_data = [
        float(request.form["sensor_1"]),
        float(request.form["sensor_2"]),
        float(request.form["sensor_3"]),
        float(request.form["sensor_4"])
    ]
    prediction = model.predict([sensor_data])[0]
    probability = model.predict_proba([sensor_data])[0][prediction] * 100

    plt.figure(figsize=(6, 4))
    sns.barplot(x=["Vibration", "Temp", "Pressure", "Voltage"], y=sensor_data)
    plt.title("Sensor Readings")
    graph_path = os.path.join("static", "graph.png")
    plt.savefig(graph_path)
    plt.close()

    return render_template("result.html", prediction=prediction, probability=round(probability, 2), graph_image=graph_path)

if __name__ == "__main__":
    app.run(debug=True)
