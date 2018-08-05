from flask import Flask
from flask_ask import Ask, statement, question
import requests
import time
import json
from camera import take_picture
import face_recognition
import numpy as np
import cv2
import os
from PIL import Image, ImageDraw
import pickle
from mynn.layers.dense import dense
from mynn.layers.conv import conv
from mynn.initializers.he_normal import he_normal
from mynn.activations.relu import relu
from mynn.activations.softmax import softmax
from mynn.losses.cross_entropy import softmax_cross_entropy 
from mynn.optimizers.sgd import SGD
from mygrad.nnet.layers import max_pool
import mygrad as mg
from mynn.initializers.glorot_uniform import glorot_uniform
from collections import defaultdict

class Model:
    #Basic model structure was needed because using the emotion models requires it
    def __init__(self):
        self.conv1 = conv(1, 10, 5, padding=0, weight_initializer=glorot_uniform)
        self.conv2 = conv(5, 20, 5, padding=0, weight_initializer=glorot_uniform)
        self.dense1 = dense(290, 20, weight_initializer=glorot_uniform)
        self.dense2 = dense(20, 2, weight_initializer=glorot_uniform)
        
    def __call__(self, x):
        x = self.conv1(x)
        x = max_pool(x, (2,2), 2)
        x = self.conv2(x)
        x = max_pool(x, (2,2), 2)
        x = relu(self.dense1(x.reshape(x.shape[0], -1)))
        return self.dense2(x)
        
    @property
    def parameters(self):
        params = []
        for layer in (self.conv1, self.conv2, self.dense1, self.dense2):
            params += list(layer.parameters)
        return params

#Load neural network model for every emotion    
angry_model = pickle.load(open('angry_model.sav', 'rb'))
fear_model = pickle.load(open('fear_model.sav', 'rb'))
happy_model = pickle.load(open('happy_model.sav', 'rb'))
Neutral_model = pickle.load(open('Neutral_model.sav', 'rb'))
sad_model = pickle.load(open('sad_model.sav', 'rb'))
surprise_model = pickle.load(open('surprise_model.sav', 'rb'))

def emotion_test(pic_test):
    '''
    Uses models to detect the emotion of a person
    Parameters:
        pic_test: A picture of a face.
    Return:
    String of one or two emotions
    '''
    allModel = [angry_model,fear_model,happy_model,sad_model,surprise_model,Neutral_model]
    allModelName = ['angry','fearful','happy','sad','surprised','neutral']
    model_dic = defaultdict(list)
    for index,i in enumerate(allModel): #Goes through all emotion models and passes the descriptors of the input image into them, then stores them in the dictionary
        try:
            encode_test = face_recognition.face_encodings(pic_test)[0].reshape(1,1,128) #Calculates the descriptors of the image using dlib
            model_dic[allModelName[index]] = mg.abs(i(encode_test)[0,0]) #Uses descriptors to calculate the emotional value for each emotion model
        except:
            print("Did not detect face")
    ansTrue = []
    model_dic['fearful'] += 0.2 #Makes fearful less common because it is detected too often
    if model_dic['angry'] < 1: #Sets the anger threshold
        ansTrue += ["angry"]
    if model_dic['sad'] < 0.9: #Sets the sadness threshold and makes sure that if sad is picked then happy will not be
        ansTrue += ["sad"]
        del model_dic['happy']
    if model_dic['fearful'] < 0.7: #Sets the fear threshold and makes sure happy is not deleted twice
        ansTrue += ["fearful"]
        try:
            del model_dic['happy']
        except:
            print("No happy exists")
    if model_dic['surprised'] < 0.5: #Sets surprise threshold
        ansTrue += ['surprised']
    if min(model_dic, key=model_dic.get) == 'neutral' and model_dic['neutral'] > 0.5: #If the chosen emotion is neutral with a value above the threshold then delete neutral
        del model_dic['neutral']
    ansTrue += [min(model_dic, key=model_dic.get)] #Add the most probable emotion
    ansTrue = list(set(ansTrue)) #Delete repeats
    return ' and '.join(ansTrue) #Join emotions with and

app = Flask(__name__)
ask = Ask(app, '/')

@app.route('/')
def homepage():
    return "Hello"

@ask.launch
def start_skill(): #Alexa says this when the skill is called
    welcome_message = 'Hello there, do you want me to predict your mood?'
    return question(welcome_message)


@ask.intent("YesIntent")
def yes(): #Alexa runs code when user agrees to run it
    pic = take_picture()
    return statement(emotion_test(pic))

@ask.intent("NoIntent")
def no(): #Alexa does not run code if user disagrees
    bye_text = 'Oooooof'
    return statement(bye_text)


if __name__ == '__main__':
    app.run(debug=True)