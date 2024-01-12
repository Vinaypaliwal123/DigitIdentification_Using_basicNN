# Number Detection Project

Hello, I'm Vinay Paliwal! This project is a simple implementation of a neural network for number detection. The neural network is implemented from scratch using numpy, pandas, and matplotlib. The goal of this project is to accurately detect numbers in images. If you find this project helpful, please consider giving it a star on GitHub!


```markdown


## Import Libraries
We start by importing the necessary libraries.
```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
```

## Load Data
Next, we load the data from a CSV file.
```python
data = pd.read_csv('train.csv')
data.head()
```

## Preprocess Data
We then preprocess the data by converting it into a numpy array, shuffling it, and splitting it into development and training sets. We also normalize the pixel values by dividing them by 255.
```python
data = np.array(data)
m, n = data.shape
# there are 42001 images which contains 784 pixel

np.random.shuffle(data)
data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
```

## Initialize Parameters
We initialize the parameters of our neural network randomly.
```python
def init_params():
    w1 = np.random.rand(10,784)-0.5
    b1 = np.random.rand(10,1)-0.5
    w2 = np.random.rand(10,10)-0.5
    b2 = np.random.rand(10,1)-0.5
    return w1,b1,w2,b2
```

## Forward Propagation
We define the forward propagation step of our neural network, which includes the ReLU and softmax activation functions.
```python
def RelU(z):
    return np.maximum(0,z)

def softmax(z):
    return np.exp(z)/sum(np.exp(z))

def forward_prop(w1,b1,w2,b2,X):
    Z1 = w1.dot(X) + b1
    A1 = RelU(Z1)
    Z2 = w2.dot(A1) +b2
    A2 = softmax(Z2)
    return Z1,A1,Z2,A2
```

## Back Propagation
We define the back propagation step of our neural network, which calculates the derivatives of the weights and biases.
```python
def One_hot(Y):
    one_hot_y = np.zeros((Y.size,Y.max()+1))
    one_hot_y[np.arange(Y.size),Y]=1
    one_hot_y = one_hot_y.T
    return one_hot_y

def deriv_RelU(Z):
    return Z>0

def back_prop(Z1,A1,Z2,A2,w1,w2,X,Y):
    m = Y.size
    one_hot_y = One_hot(Y)
    dZ2 = A2-one_hot_y
    dw2 =1/m*dZ2.dot(A1.T)
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = w2.T.dot(dZ2)*deriv_RelU(Z1)
    dw1 =1/m*dZ1.dot(X.T)
    db1 =1/m * np.sum(dZ1, axis=1, keepdims=True)
    return dw1,db1,dw2,db2
```

## Update Parameters
We update the parameters of our neural network using the derivatives calculated in the back propagation step.
```python
def update_params(w1,b1,w2,b2,dw1,db1,dw2,db2,alpha):
    w1 = w1 - alpha*dw1
    b1 = b1 - alpha*db1
    w2 = w2 - alpha*dw2
    b2 = b2 - alpha*db2
    return w1,b1,w2,b2
```

## Apply Gradient Descent Algorithm
We apply the gradient descent algorithm to train our neural network. The algorithm iteratively performs forward propagation, back propagation, and parameter updates.
```python
def get_predictions(A2):
    return np.argmax(A2,0) #axis = 0 means A2 will be in y axis form
def get_accuracy(predictions,Y):
    print(predictions,Y)
    return np.sum(predictions == Y)/Y.size

def gradient_descent(X,Y,alpha,iterations):
    accuracy =0
    w1,b1,w2,b2 = init_params()
    for i in range(iterations):
        Z1,A1,Z2,A2 = forward_prop(w1,b1,w2,b2,X)
        dw1,db1,dw2,db2 = back_prop(Z1,A1,Z2,A2,w1,w2,X,Y)
        w1,b1,w2,b2 = update_params(w1,b1,w2,b2,dw1,db1,dw2,db2,alpha)
        
        if i%10 == 0:
            print('Iteration: ', i)
            predictions = get_predictions(A2)
            accuracy = get_accuracy(predictions,Y)
            print(accuracy)

    return w1,b1,w2,b2,accuracy
```

## Make Predictions
We use the trained parameters of our neural network to make predictions on new data.
```python
w1,b1,w2,b2,accuracy = gradient_descent(X_train,Y_train,0.10,500)
```

## Test Predictions
We test the predictions of our neural network on some images in our training set.
```python
def make_predictions(X,w1,b1,w2,b2):
    _,_,_,A2 = forward_prop(w1,b1,w2,b2,X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index,w1,b1,w2,b2):
    current_image = X_train[:,index,None]
    prediction = make_predictions(current_image,w1,b1,w2,b2)
    label = Y_train[index]
    print("prediction: ", prediction)
    print("label: ",label)
    current_image = current_image.reshape((28, 28)) * 255.
    plt.gray()
    plt.imshow(current_image,interpolation = 'nearest')
    plt.show()

test_prediction(0,w1,b1,w2,b2)
test_prediction(1,w1,b1,w2,b2)
test_prediction(2,w1,b1,w2,b2)
test_prediction(3,w1,b1,w2,b2)
```

## Accuracy
Finally, we calculate the accuracy of our neural network in percentage.
```python
accuracy*100
```
```
