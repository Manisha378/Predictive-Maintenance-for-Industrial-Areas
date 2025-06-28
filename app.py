from flask import Flask, render_template, request
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)
model = joblib.load("models/final_model.pkl")

@app.route("/")
def login():
    return render_template("login.html")

@app.route("/upload")
def upload():
    return render_template("upload.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/result", methods=["POST"])
def result():
    data = [
        float(request.form["sensor_1"]),
        float(request.form["sensor_2"]),
        float(request.form["sensor_3"]),
        float(request.form["sensor_4"])
    ]
    prediction = model.predict([data])[0]
    probability = model.predict_proba([data])[0][prediction] * 100

    plt.figure()
    sns.barplot(x=["Sensor 1", "Sensor 2", "Sensor 3", "Sensor 4"], y=data)
    plt.title("Sensor Data Visualization")
    path = os.path.join("static", "graph.png")
    plt.savefig(path)
    plt.close()

    return render_template("result.html", prediction=prediction, probability=round(probability, 2), graph_image=path)

if __name__ == "__main__":
    app.run(debug=True)