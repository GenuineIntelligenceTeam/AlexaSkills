"""

from flask import *
app = Flask(__name__)
@app.route('/submit', methods=['GET', 'POST'])
def submit():
    return render_template('display.html', color=request.form['color'])

@app.route('/')
def index():
  return render_template('form.html')
app.run(debug=True)

"""


from flask import Flask
from flask_ask import Ask, statement, question
import json
import random
from PIL import Image
import sys
import time
import io
import numpy as np
from camera import take_picture


app = Flask(__name__)
ask = Ask(app, '/')

@app.route('/')
def homepage():
    return "Hello"


@ask.launch
def start_skill():
    welcome_message = 'Would You Like To Play Intuition?'
    return question(welcome_message)

def isBlack():
    time.sleep(3)
    pic = take_picture()
    i = Image.fromarray(pic)
    h = i.histogram()

    # split into red, green, blue
    r = h[0:256]
    g = h[256:256*2]
    b = h[256*2: 256*3]

    # perform the weighted average of each channel:
    # the *index* is the channel value, and the *value* is its weight

    red=    sum( i*w for i, w in enumerate(r) ) / sum(r)
    green = sum( i*w for i, w in enumerate(g) ) / sum(g)
    blue = sum( i*w for i, w in enumerate(b) ) / sum(b)

    if (int(red) < 50):
        if(int(green)< 50):
            if(int(blue)<50):
                return True
            else: return False
        else: return False
    else: return False

def evaluate (black_boolean):
  if(black_boolean == False):
      return statement("Nope")
  if(black_boolean):
    loop = False
    return ""

def say_take_pic():
  return statement("Hold Up The Object")

def say_no():
  return statement("Nope")


@ask.intent("YesIntent")
def yes():
  loop = False
  while(loop == True):
    say_take_pic()
    black_boolean = isBlack()
    evaluate(black_boolean)
  isBlack()
  say_no()
  return statement("That's Your Object")

@ask.intent("NoIntent")
def no():
  bye_text = "Okay. Nevermind"
  return statement(bye_text)


if __name__ == '__main__':
    app.run(debug=True)