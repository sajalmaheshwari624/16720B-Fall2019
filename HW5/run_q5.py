import numpy as np
import scipy.io
from nn import *
from collections import Counter
import matplotlib.pyplot as plt

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36 
learning_rate =  3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

# Q5.1 & Q5.2
# initialize layers here
initialize_weights(1024,32,params, 'layer1')
initialize_weights(32,32,params,'layer2')
initialize_weights(32,32,params,'layer3')
initialize_weights(32,1024,params,'decode1')

layerCount = len(params) // 2 - 1
# should look like your previous training loops
trainLossEpoch = np.zeros(max_iters)
validLossEpoch = np.zeros(max_iters)
for itr in range(max_iters):
    total_loss = 0
    for xb,_ in batches:
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions
        e1 = forward(xb, params, 'layer1', activation = relu)
        e2 = forward(e1, params, 'layer2', activation = relu)
        d2 = forward(e2, params, 'layer3', activation = relu)
        output = forward(d2, params, 'decode1')

        loss = np.sum(np.square(output - xb))
        total_loss += loss
        delta1 = -2 * (xb - output)
        delta2 = backwards(delta1,params,'decode1', sigmoid_deriv)
        delta3 = backwards(delta2,params,'layer3', relu_deriv)
        delta4 = backwards(delta3,params,'layer2', relu_deriv)
        backwards(delta4, params, 'layer1', relu_deriv)
        for index in range(layerCount) :
            layerName = 'layer' + str(index + 1)
            #print (layerName)
            params['m_W' + layerName] = 0.9*params['m_W' + layerName] - learning_rate*params['grad_W' + layerName]
            params['m_b' + layerName] = 0.9*params['m_b' + layerName] - learning_rate*params['grad_b' + layerName]
            params['W' + layerName] += params['m_W' + layerName]            

        params['m_Wdecode1'] = 0.9*params['m_Wdecode1'] - learning_rate*params['grad_Wdecode1']
        params['m_bdecode1'] = 0.9*params['m_bdecode1'] - learning_rate*params['grad_bdecode1']
        params['Wdecode1'] += params['m_Wdecode1']

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9

    e1_valid = forward(valid_x, params, 'layer1', activation = relu)
    e2_valid = forward(e1_valid, params, 'layer2', activation = relu)
    d2_valid = forward(e2_valid, params, 'layer3', activation = relu)
    output_valid = forward(d2_valid, params, 'decode1')
    valid_loss = np.sum(np.square(output_valid - valid_x))
    trainLossEpoch[itr] = total_loss / train_x.shape[0]
    validLossEpoch[itr] = valid_loss / valid_x.shape[0]

plt.figure('loss')
plt.plot(range(max_iters), trainLossEpoch)
plt.plot(range(max_iters), validLossEpoch)
plt.legend(['train', 'validation'])
plt.show()
        
# Q5.3.1
import matplotlib.pyplot as plt
e1_valid = forward(valid_x, params, 'layer1', activation = relu)
e2_valid = forward(e1_valid, params, 'layer2', activation = relu)
d2_valid = forward(e2_valid, params, 'layer3', activation = relu)
output_valid = forward(d2_valid, params, 'decode1')

indices = np.random.permutation(10)
indices = [14, 57, 489, 465, 1730, 1720, 2020, 2050, 3450, 3499]
for index in indices :
    xInput = valid_x[index]
    xInput = np.reshape(xInput, (32, 32))
    xInput = np.transpose(xInput)

    xOutput = output_valid[index]
    xOutput = np.reshape(xOutput, (32, 32))
    xOutput = np.transpose(xOutput)

    plt.imshow(xInput)
    plt.show()

    plt.imshow(xOutput)
    plt.show()

from skimage.measure import compare_psnr as psnr
# evaluate PSNR
e1_valid = forward(valid_x, params, 'layer1', activation = relu)
e2_valid = forward(e1_valid, params, 'layer2', activation = relu)
d2_valid = forward(e2_valid, params, 'layer3', activation = relu)
output_valid = forward(d2_valid, params, 'decode1')

valPSNR = 0
for i in range(valid_x.shape[0]) :
    real_out = valid_x[i]
    pred_out = output_valid[i]
    valPSNR += psnr(real_out, pred_out)
print (valPSNR / valid_x.shape[0])

