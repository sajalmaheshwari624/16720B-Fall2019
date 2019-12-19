import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt


train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']
max_iters = 75

# pick a batch size, learning rate
batch_size = 32
learning_rate = 0.002
hidden_size = 64

#print (train_x.shape)
batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}
initialize_weights(1024,hidden_size,params,'layer1')
initialize_weights(hidden_size,36,params,'output')

init_hidden_weights = params['Wlayer1']

# with default settings, you should get loss < 150 and accuracy > 80%
trainLossEpoch = np.zeros(max_iters)
validLossEpoch = np.zeros(max_iters)
trainAccuEpoch = np.zeros(max_iters)
validAccuEpoch = np.zeros(max_iters)

for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    count = 0
    for xb,yb in batches:
        count += 1
        h1 = forward(xb,params,'layer1')
        probs = forward(h1,params,'output',softmax)
        loss,acc = compute_loss_and_acc(yb, probs)
        total_loss +=  loss
        total_acc += acc
        delta1 = probs - yb
        delta2 = backwards(delta1,params,'output',linear_deriv)
        backwards(delta2,params,'layer1',sigmoid_deriv)

        WHidden = params['Wlayer1']
        WOut = params['Woutput']

        biasHidden = params['blayer1']
        biasOut = params['boutput']

        params['Wlayer1'] = WHidden - learning_rate * params['grad_Wlayer1']
        params['Woutput'] = WOut - learning_rate * params['grad_Woutput']
        params['blayer1'] = biasHidden - learning_rate * params['grad_blayer1']
        params['boutput'] = biasOut - learning_rate * params['grad_boutput']

    total_acc = total_acc / batch_num
    total_loss = total_loss / train_x.shape[0]
    trainLossEpoch[itr] = total_loss
    trainAccuEpoch[itr] = total_acc

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))

    
    h1_valid = forward(valid_x,params,'layer1')
    probs_valid = forward(h1_valid,params,'output',softmax)
    loss_valid,acc_valid = compute_loss_and_acc(valid_y, probs_valid)
    validLossEpoch[itr] = loss_valid / valid_x.shape[0]
    validAccuEpoch[itr] = acc_valid

plt.figure('accuracy')
plt.plot(range(max_iters), trainAccuEpoch)
plt.plot(range(max_iters), validAccuEpoch)
plt.legend(['train', 'validation'])
plt.show()

plt.figure('loss')
plt.plot(range(max_iters), trainLossEpoch)
plt.plot(range(max_iters), validLossEpoch)
plt.legend(['train', 'validation'])
plt.show()

        


# run on validation set and report accuracy! should be abov
h1_valid = forward(valid_x,params,'layer1')
probs_valid = forward(h1_valid,params,'output',softmax)
loss_valid,valid_acc = compute_loss_and_acc(valid_y, probs_valid)
print('Validation accuracy: ',valid_acc)

h1_test = forward(test_x,params,'layer1')
probs_test = forward(h1_test,params,'output',softmax)
loss_test,test_acc = compute_loss_and_acc(test_y, probs_test)
print('Test accuracy: ',test_acc)

if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
    hiddenWeights = saved_params['Wlayer1']
# Q3.1.3
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

fig = plt.figure()
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(8, 8),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )

for i in range(hiddenWeights.shape[1]):
    layerWeight = hiddenWeights[:,i]
    layerImage = np.reshape(layerWeight, (32, 32))
    grid[i].imshow(layerImage)

plt.show()

fig = plt.figure()
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(8, 8),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )

for i in range(init_hidden_weights.shape[1]):
    layerWeight = init_hidden_weights[:,i]
    layerImage = np.reshape(layerWeight, (32, 32))
    grid[i].imshow(layerImage)

plt.show()

# Q3.1.4
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))

# compute comfusion matrix here
h1_test = forward(test_x,params,'layer1')
probs_test = forward(h1_test,params,'output',softmax)
trueLabel = np.argmax(test_y, axis = 1)
predLabel = np.argmax(probs_test, axis = 1)
for i in range(trueLabel.size) :
    confusion_matrix[trueLabel[i]][predLabel[i]] += 1

import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()

h1_test = forward(train_x,params,'layer1')
probs_test = forward(h1_test,params,'output',softmax)
trueLabel = np.argmax(train_y, axis = 1)
predLabel = np.argmax(probs_test, axis = 1)
for i in range(trueLabel.size) :
    confusion_matrix[trueLabel[i]][predLabel[i]] += 1

plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()

h1_test = forward(valid_x,params,'layer1')
probs_test = forward(h1_test,params,'output',softmax)
trueLabel = np.argmax(valid_y, axis = 1)
predLabel = np.argmax(probs_test, axis = 1)
for i in range(trueLabel.size) :
    confusion_matrix[trueLabel[i]][predLabel[i]] += 1

plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()