# libraries
import random
import numpy as np
from flask_cors import CORS
import pickle
import json
from flask import Flask, render_template, request, jsonify
# from flask_ngrok import run_with_ngrok
from keras.models import load_model
from utils import get_response, predict_class

# chat initialization
model = load_model("chatbot_model.h5")
intents = json.loads(open("intents.json").read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

app = Flask(__name__)
CORS(app)


# run_with_ngrok(app)

# ---------------------------------------- ENDPOINTS ----------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get", methods=["POST"])
def chatbot_response():
    msg = request.form["msg"]
    print(msg)
    if msg.startswith('my name is'):
        name = msg[11:]
        ints = predict_class(msg, model)
        res1 = get_response(ints, intents)
        res = res1.replace("{n}", name)
    elif msg.startswith('hi my name is'):
        name = msg[14:]
        ints = predict_class(msg, model)
        res1 = get_response(ints, intents)
        res = res1.replace("{n}", name)
    else:
        ints = predict_class(msg, model)
        res = get_response(ints, intents)

    return res


@app.route("/get_response", methods=["POST"])
def chatbot_response1():

    msg = request.form["msg"]
    if msg.startswith('my name is'):
        name = msg[11:]
        ints = predict_class(msg, model)
        res1 = get_response(ints, intents)
        res = res1.replace("{n}", name)
    elif msg.startswith('hi my name is'):
        name = msg[14:]
        ints = predict_class(msg, model)
        res1 = get_response(ints, intents)
        res = res1.replace("{n}", name)
    else:
        ints = predict_class(msg, model)
        res = get_response(ints, intents)

    if res:
        response = jsonify(data=res, success=True)
        # Enable Access-Control-Allow-Origin
        # response.headers.add("Access-Control-Allow-Origin", "*")
        return response

    response = jsonify(data='None', success=False)
    return response


if __name__ == "__main__":
    app.run(debug=True)
