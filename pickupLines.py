from flask import Flask
from flask_ask import Ask, statement, question
import requests
import time
import unidecode
import json
import random
import numpy as np

app = Flask(__name__)
ask = Ask(app, '/')

@app.route('/')
def homepage():
    return "Hello"

@ask.launch
def start_skill():
    welcome_message = 'Hello there, I\'m guessing you\'re in need of some good pickup lines?'
    return question(welcome_message)


lines = ["Is your name Google? Because you have everything I\'ve been searching for.",
"Is your name Wi-fi? Because I\'m really feeling a connection.",
"You had me at \"Hello World.\"",
"Hi, my name\'s Microsoft. Can I crash at your place tonight?",
"Are you a computer keyboard? Because you\'re my type.",
"You auto-complete me."]


@ask.intent("YesIntent")
def yes():
    l = np.random.randint(0,5)
    return statement(lines[l])

@ask.intent("NoIntent")
def no():
    bye_text = 'Okay, I tried'
    return statement(bye_text)


if __name__ == '__main__':
    app.run(debug=True)
