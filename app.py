# libraries
from flask_cors import CORS
import pickle
import json
from flask import Flask, render_template, request, jsonify
from keras.models import load_model

# chat initialization
model = load_model("chatbot_model.h5")
intents = json.loads(open("intents.json").read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

app = Flask(__name__)
CORS(app)

# run_with_ngrok(app)
import nltk
# nltk.download('all')
import random
import numpy as np
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import pickle
import json

lemmatizer = WordNetLemmatizer()
# chat initialization
# model = load_model("chatbot_model.h5")
# intents = json.loads(open("intents.json").read())
# words = pickle.load(open("words.pkl", "rb"))
# classes = pickle.load(open("classes.pkl", "rb"))


# chat functionalities
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)


def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def get_response(ints, intents_json):
    tag = ints[0]["intent"]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result


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
    app.run()
