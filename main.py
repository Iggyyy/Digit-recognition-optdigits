import numpy as np
from readcsv import GetData
#import
np.random.seed(1)
images, labels, t_images, t_labels = GetData.imgs, GetData.labels, GetData.t_imgs, GetData.t_labels

DATAPOINTS = len(images)
DP_SHAPE = len(images[0])
HIDDEN_SIZE = 50
ITERATIONS  = 500
BATCH_SIZE = 20
LABELS_SIZE = len(labels[0])
ALPHA = 0.05
#TODO implement proper softmax and tanh
w_01 = 0.2 * np.random.random((DP_SHAPE, HIDDEN_SIZE)) - 0.1
w_12 = 0.2 * np.random.random((HIDDEN_SIZE, LABELS_SIZE)) - 0.1

def tanh(x):
    return np.tanh(x)
def tanh_deriv(x):
    return 1 - (x**2)
def softmax(x):
    z = x - np.max(x)
    tmp = np.exp(z)
    #print(tmp)
    return tmp / np.sum(tmp, axis=1, keepdims=True)

def relu(x):
    return x * (x>=0)
def relu_deriv(x):
    return x >= 0

for it in range(ITERATIONS):

    correct_cnt = 0
    error = 0.0

    for i in range(int(DATAPOINTS/BATCH_SIZE)):

        b_start, b_end = (BATCH_SIZE*i, BATCH_SIZE*(i+1))

        lay_0 = images[b_start:b_end]
        lay_1 = tanh(np.dot(lay_0, w_01))

        drop_mask =  np.random.randint(2, size = lay_1.shape)
        lay_1 *= drop_mask * 2

        lay_2 = np.dot(lay_1, w_12)

        error += np.sum( (labels[b_start:b_end] - lay_2) ** 2 )

        for k in range(BATCH_SIZE):
            correct_cnt += int(  np.argmax(lay_2[k:k+1]) == np.argmax(labels[b_start + k: b_start + k + 1]) )

        delta_2 =  (( labels[b_start: b_end] - lay_2 ) / ( BATCH_SIZE * lay_2.shape[0] ) )

        delta_1 = np.dot(delta_2, w_12.T) * tanh_deriv(lay_1)

        delta_1 *= drop_mask

        
        w_12 += ALPHA * np.dot(lay_1.T, delta_2)
        w_01 += ALPHA * np.dot(lay_0.T, delta_1)

        
        
    
    if it % 10 == 0 or it == ITERATIONS - 1:
        #print(error)
        t_correct_cnt  = 0
        t_error = 0

        for k in range(len(t_labels)):

            lay_0 = t_images[k:k+1]
            lay_1 = relu(np.dot(lay_0, w_01))
            lay_2 = np.dot(lay_1, w_12)

            t_error += np.sum( (t_labels[k:k+1] - lay_2) ** 2 )
            t_correct_cnt += int(  np.argmax(lay_2) == np.argmax(t_labels[k:k + 1]) )

        print(it, ": Train_acc: " + str(correct_cnt/DATAPOINTS)[0:5] + " Train_err: " + str(error / DATAPOINTS)[0:5] + " || Test_acc: " + str(t_correct_cnt/ len(t_labels))[0:5] )





