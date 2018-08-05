'''
    'study_mode.py' uses FlaskAsk to program the Alexa Skill "Study Mode". In this
    skill, Alexa will monitor whether the user closes their eyes for too long or
    falls asleep, and whether the room is too loud for a suitable studying
    environment.
    This Alexa Skill is a feature of the 'Study Buddy'.
    _________________________________________________________
    By the Genuine Intelligence Team of BWSI Cog*Works 2018
'''

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
    welcome_message = 'Please state the status'
    return question(welcome_message)


@ask.intent("loud")
def loud():
    loud = 'It sounds like the room is too loud. Go find a quieter place to study.'
    return statement(loud)

@ask.intent("distracting")
def distracting():
    output = 'It sounds like people are chatting. Stay focused.'
    return statement(output)

@ask.intent("sleep")
def sleep():
    wake_up = 'Hello there, please wake up, we\'re on study mode!'
    return statement(wake_up)

@ask.intent("StopIntent")
def sleep():
    bye_text = 'Study Mode turned off'
    return statement(bye_text)


if __name__ == '__main__':
    app.run(debug=True)
