import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

############################## Q 2.1 ##############################
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    W, b = None, None

    ##########################
    ##### your code here #####
    ##########################
    b = np.zeros(out_size)
    var = 2 / (in_size + out_size)
    dem = (in_size + out_size)
    low = -1 * np.sqrt(6 / dem)
    high = np.sqrt(6 / dem)
    W = np.random.uniform(low = low, high = high, size = (in_size, out_size))
    params['W' + name] = W
    params['b' + name] = b

############################## Q 2.2.1 ##############################
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    res = None

    ##########################
    ##### your code here #####
    ##########################
    res = 1 / (1 + np.exp(-x))
    return res

############################## Q 2.2.1 ##############################
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]


    ##########################
    ##### your code here #####
    ##########################
    #print ("Shape of input", X.shape)
    #print ("Shape of W", W.shape)
    pre_act = np.matmul(X, W) + b
    post_act = activation(pre_act)

    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

############################## Q 2.2.2  ##############################
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    res = None

    ##########################
    ##### your code here #####
    ##########################
    maxVal = np.max(x, axis = 1)
    c = -maxVal
    cRes = np.reshape(c, (c.size,1))
    cMat = np.tile(cRes, x.shape[1])
    xStable = x - cMat
    expXVal = np.exp(xStable)
    expSum = np.sum(expXVal, axis = 1)
    expSumRes = np.reshape(expSum,(expSum.size,1))
    expMat = np.tile(expSumRes, x.shape[1])
    res = np.divide(expXVal, expMat)
    return res

############################## Q 2.2.3 ##############################
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    loss, acc = None, None

    logProbs = np.log(probs)
    termToSum = np.multiply(y, logProbs)
    loss = -1*np.sum(termToSum[:])

    predLabels = np.argmax(probs, axis = 1)
    trueLabels = np.argmax(y, axis = 1)

    labelDiff = predLabels - trueLabels
    count = 0
    for diff in labelDiff :
        if diff == 0 :
            count += 1
    #print (count)
    totalNum = probs.shape[0]
    acc = count / totalNum
    return loss, acc 

############################## Q 2.3 ##############################
# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]
    grad_X, grad_W, grad_b = np.zeros(X.shape), np.zeros(W.shape), np.zeros(b.shape)
    # do the derivative through activation first
    # then compute the derivative W,b, and X
    ##########################
    ##### your code here #####
    ##########################
    #print (delta.shape)
    #print (W.shape)
    delta_post_act = activation_deriv(post_act)
    delta_pre_act = np.multiply(delta, delta_post_act)

    for i in range(X.shape[0]) :
        xVal = X[i,:].reshape(X[i,:].size,1)
        pre_act_val = delta_pre_act[i,:].reshape(1, delta_pre_act[i,:].size)
        grad_W += np.matmul(xVal, pre_act_val)
        grad_b += delta_pre_act[i,:]

        grad_X[i,:] = np.matmul(W, np.transpose(pre_act_val)).reshape([-1])

    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

############################## Q 2.4 ##############################
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    batches = []
    ##########################
    ##### your code here #####
    ##########################
    num_examples = y.shape[0]
    randomIndex = np.random.permutation(num_examples)
    batches = []
    for i in range(0, num_examples, 5) :
        indices = randomIndex[i:i+5]
        batch_x = x[indices,:]
        batch_y = y[indices,:]
        batches.append((batch_x, batch_y))
    return batches
