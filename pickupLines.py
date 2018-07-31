from flask import Flask
from flask_ask import Ask, statement, question
import requests
import time
import unidecode
import json
import random

app = Flask(__name__)
ask = Ask(app, '/')

@app.route('/')
def homepage():
    return "Hello"

@ask.launch
def start_skill():
    welcome_message = 'Hello there, I\'m guessing you\'re in need of some good pickup lines?'
    return question(welcome_message)

def generateLine():
    lines = ["Is your name Google? Because you have everything I\'ve been searching for.",
    "Is your name Wi-fi? Because I\'m really feeling a connection.",
    "You had me at \"Hello World.\"",
    "Hi, my name\'s Microsoft. Can I crash at your place tonight?",
    "Are you a computer keyboard? Because you\'re my type.",
    "You auto-complete me."]
    l = np.random.randint(0,5)
    return lines, l

@ask.intent("YesIntent")
def yes():
    lines, l = generateLine()
    more = "Do you want more?"
    return statement(lines[l]), question(more)

@ask.intent("NoIntent")
def no():
    bye_text = 'Okay, I tried'
    return statement(bye_text)


if __name__ == '__main__':
    app.run(debug=True)
