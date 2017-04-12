import math
import csv
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split




random.seed(0)

def rand(a,b):
    return (b-a)*random.random() + a

def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def dsigmoid(y):
    return y*(1-y)




def loadcsv(filename):
    with open ( filename, "r" ) as csvfile:
        lines = csv.reader ( csvfile )
        dataset = list ( lines )
        for i in range ( len ( dataset ) ):
            dataset[ i ] = [ float ( x ) for x in dataset[ i ] ]
        return dataset

class NN:
    def __init__(self, ni, nh, no):

        self.ni = ni + 1
        self.nh = nh
        self.no = no

        #activation function
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no

        #weight matrix
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)

        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-1.0, 1.0)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-1.0, 1.0)


    def update(self, inputs):
        if len(inputs) != self.ni-1:
            raise ValueError('wrong number of inputs')

        # input activations
        for i in range(self.ni-1):
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)

        # output activations
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao


    def backPropagate(self, targets, N):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k]-self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        # error terms for hidden
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k]*self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k]*self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N*change


        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j]*self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change



        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k]-self.ao[k])**2
        return error

    def get_labels(self, patterns):
        data_len= len(patterns)
        num_classes= self.no

        labels= makeMatrix(data_len,num_classes)
        for i, instance in enumerate(patterns):
            label_value= int(instance[-1][0])
            labels[i][label_value-1]= 1.0

        return labels

    def test(self, patterns):
        total=0
        correct=0
        labels= self.get_labels(patterns)
        for i,p in enumerate(patterns):
            total= total+1
            answer= self.update(p[0])
            answer= np.argmax(answer)
            req_answer= np.argmax(labels[i])
            if(answer==req_answer):
                correct=correct+1

            print(p, '->', self.update(p[0]))

        print("accuracy : ")
        print(correct/total)

    # def weights(self):
    #     print('Input weights:')
    #     for i in range(self.ni):
    #         print(self.wi[i])
    #     print()
    #     print('Output weights:')
    #     for j in range(self.nh):
    #         print(self.wo[j])

    def train(self, patterns, iterations=1000, N=0.5):
        labels= self.get_labels(patterns)
        for k in range(iterations):
            error = 0.0
            old_error=0.0
            for i,p in enumerate(patterns):
                inputs = p[0]
                label= labels[i] #resume from here.
                self.update(inputs)
                error = error + self.backPropagate(label, N) #change the backpropagation function
            if k % 100 == 0:
                print('error %-.5f' % error)
            if error< 0.0005:
                break





def demo():
    filename = 'creditcard.csv'
    dataset = loadcsv ( filename )
    train, test = train_test_split ( dataset, test_size=0.3 )

    pat_train=[]
    for i in range ( len ( train ) ):
        label = [ ]
        label.append ( train[ i ][ -1 ] )
        features = train[ i ][ :-1 ]
        temp = [ ]
        temp.append ( features )
        temp.append ( label )
        pat_train.append ( temp )


    pat_test = [ ]
    for i in range ( len ( test ) ):
        label = [ ]
        label.append ( test[ i ][ -1 ] )
        features = test[ i ][ :-1 ]
        temp = [ ]
        temp.append ( features )
        temp.append ( label )
        pat_test.append ( temp )

    # input_neuron= int(input("give the number of input neurons"))
    # hidden_neuron= int(input("give the number of hidden neurons"))
    # output_neuron= int(input("give the number of classes"))

    n = NN(9,7,2)

    n.train(pat_train)

    n.test(pat_test)


if __name__ == '__main__':
    demo()