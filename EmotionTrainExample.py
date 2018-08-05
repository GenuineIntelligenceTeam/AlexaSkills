import audioscrape
import urllib
import time
import pickle
import shutil
from requests import get
import numpy as np
from camera import take_picture
import face_recognition
import numpy as np
import cv2
import os
from mynn.layers.dense import dense
from mynn.layers.conv import conv
from mynn.initializers.he_normal import he_normal
from mynn.activations.relu import relu
from mynn.activations.softmax import softmax
from mynn.losses.cross_entropy import softmax_cross_entropy 
from mynn.optimizers.sgd import SGD
from mygrad.nnet.layers import max_pool
from mynn.initializers.glorot_uniform import glorot_uniform

curdir = os.path.abspath(os.path.dirname("./Pictures"))
filename = './Pictures/fer2013.csv' #Directory of Kaggle data
filename = os.path.join(curdir,filename)
csvfile = filename
channel = 1
data = pd.read_csv(csvfile,delimiter=',',dtype='a') 
labels = np.array(data['emotion'],np.float) #Labels for each image
imagebuffer = np.array(data['pixels'])
images = np.array([np.fromstring(image,np.uint8,sep=' ') for image in imagebuffer]) #All images
del imagebuffer
num_shape = int(np.sqrt(images.shape[-1]))
images.shape = (images.shape[0],num_shape,num_shape)

def accuracy(predictions, truth):
    """
    Returns the mean classification accuracy for a batch of predictions.
    
    Parameters
    ----------
    predictions : Union[numpy.ndarray, mg.Tensor], shape=(M, D)
        The scores for D classes, for a batch of M data points
    truth : numpy.ndarray, shape=(M,)
        The true labels for each datum in the batch: each label is an
        integer in [0, D)
    
    Returns
    -------
    float
    """
    if isinstance(predictions, mg.Tensor):
        predictions = predictions.data
    return np.mean(np.argmax(predictions, axis=1) == truth)

#Gets the descriptors for each image, this may take a while so save the encodedArr variable later rather than rerun this 
N = images.shape[0]
encodedArr = np.zeros((N,128))
indexToRemove = [] #Holds the index of the images whose face could not be found
for i in range(0,N):
    backtorgb = cv2.cvtColor(images[i],cv2.COLOR_GRAY2RGB) #Turns image to RGB
    try:
        encodedArr[i] = face_recognition.face_encodings(backtorgb)[0] #Gets the descriptors for each image
    except IndexError:
        indexToRemove += [i]

new_encodedArr = np.delete(encodedArr,indexToRemove,axis=0) #Removes bad images
new_labels = np.asarray(np.delete(labels,indexToRemove),dtype=np.int) #Removes labels of bad images

#Here change all the labels to match the emotion for this specific neural network so for anger you would do the following (turn all none anger emotions to 1) 
#0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
new_labels[new_labels!=0] = 1

#Sets up test and train data
x_train = new_encodedArr[0:23796]
y_train = new_labels[0:23796]
x_test = new_encodedArr[23796:24796]
y_test = new_labels[23796:24796]

#Reshape the data to fit the network
x_train = x_train.reshape(23796, 1, 128)
x_test = x_test.reshape(1000, 1, 128)


class Model:
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
    
#Initialize model
model = Model()
optim = SGD(model.parameters, learning_rate=0.1, momentum=0.9, weight_decay=5e-04)

#Training starts here
batch_size = 100
for epoch_cnt in range(15):
    idxs = np.arange(len(x_train))
    np.random.shuffle(idxs)  
    
    
    for batch_cnt in range(0, len(x_train)//batch_size):
        batch_indices = idxs[batch_cnt*batch_size : (batch_cnt + 1)*batch_size]
        batch = x_train[batch_indices] 

        # compute the predictions for this batch by calling on model
        prediction = model(batch)
        # compute the true (a.k.a desired) values for this batch: 
        true = y_train[batch_indices]
        # compute the loss associated with our predictions(use softmax_cross_entropy)
        loss = softmax_cross_entropy(prediction,true)
        # back-propagate through your computational graph through your loss
        loss.backward()
        # compute the accuracy between the prediction and the truth 
        acc = accuracy(prediction,true)
        # execute gradient descent by calling step() of `optim`
        optim.step()
        # null your gradients (please!)
        loss.null_gradients()
    
    # Here, we evaluate our model on batches of *testing* data
    # this will show us how good our model does on data that 
    # it has never encountered
    # Iterate over batches of *testing* data
    for batch_cnt in range(0, len(x_test)//batch_size):
        idxs = np.arange(len(x_test))
        batch_indices = idxs[batch_cnt*batch_size : (batch_cnt + 1)*batch_size]
        batch = x_test[batch_indices] 
        
        # get your model's prediction on the test-batch
        test_pred = model(batch)
        # get the truth values for that test-batch
        test_true = y_test[batch_indices]
        # compute the test accuracy
        test_acc = accuracy(test_pred,test_true) 

#Test your model
pic_test = take_picture() #Takes picture
encode_test = face_recognition.face_encodings(pic_test)[0].reshape(1,1,128) #Converts image to descriptor
print((model(encode_test)))

#Save the model, change filename based on emotion
filename = 'angry_model.sav'
pickle.dump(model, open(filename, 'wb'))