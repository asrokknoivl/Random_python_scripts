import numpy as np
import pandas as pd
import copy
from math import exp

class Perceptron:
    def __init__(self, num_of_weights):
        self.W = np.array(np.random.rand(1,num_of_weights))
        self.b = np.random.rand(1)
        self.num_of_weights = num_of_weights
        

    def fit_limited(self, *X_true, y_true, alpha, epochs, trials):
        max_accs = 0
        ultm_w = 0
        ultm_b = 0
        for i,trial in enumerate(range(trials)):
            accs = 0
            best_w = 0
            best_b = 0
            print(f'Trial {i}; Initial weights: {self.W[0]}; initial bias: {self.b}')
            for epoch in range(epochs):
                pred = []
                for x, y in zip(X,y_true):
                    y_pred = (self.W * x).sum() + self.b  
                    y_step = 1 if y_pred >= 0 else 0   
                    pred.append(y_step)
                    if y != y_step:
                        err = y - y_step
                        j = 0
                        while j < len(self.W[0]):
                            delta_w = x * err * alpha
                            self.W[0][j] += delta_w[j]
                            self.b += err * alpha
                            j += 1
                new_accs = int(round(self.accuracy(y_true, pred), 2)*100)
                if new_accs >= accs:
                    accs = new_accs
                    best_w = copy.deepcopy(self.W[0])
                    best_b = copy.deepcopy(self.b)
                print(f'Trial {i}; Epoch {epoch}; weights: {self.W}; Bias: {self.b}; Training Accuracy: {new_accs}%')   
            if accs >= max_accs:
                max_accs = accs
                ultm_w = best_w
                ultm_b = best_b
            print(f'Trial {i}; highest accuracy reached was {accs}% for the values {best_w} as the Weights(W) and {best_b} as the Bias(b)')
            self.W = np.array(np.random.rand(1,self.num_of_weights))
            self.b = np.random.rand(1)
        print('.........FINISHED..............')
        print(f'Maximum accuracy reached through all trials and epochs is {max_accs}% for the values {ultm_w} as the weights(W) and {ultm_b} as the bias(b)')
        return ultm_w

    def fit_convergence(self, *X_true, y_true, alpha, conv, trials):
        acc = 0
        j = 0
        conv_c = 0
        max_accs = 0
        ultm_w = 0
        ultm_b = 0
        for k,trial in enumerate(range(trials)):
            accs = 0
            best_w = 0
            best_b = 0
            epoch = 0
            print(f'Initial weights: {self.W[0]}; initial bias: {self.b}')
            while True:
                pred = []
                for x, y in zip(X,y_true):
                    y_pred = (self.W * x).sum() + self.b  
                    y_step = self.step(y_pred)   
                    pred.append(y_step)
                    if y != y_step:
                        err = y - y_step
                        i = 0
                        while i < len(self.W[0]):
                            delta_w = x * err * alpha
                            self.W[0][i] += delta_w[i]
                            self.b += err * alpha
                            i += 1
                new_accs = int(round(self.accuracy(y_true, pred), 2)*100)
                if new_accs >= accs:
                    accs = new_accs
                    best_w = copy.deepcopy(self.W[0])
                    best_b = copy.deepcopy(self.b)
                print(f'Trial {k}; Epoch {epoch}; weights: {self.W}; Bias: {self.b}; Training Accuracy: {new_accs}%')   
                if new_accs in range(acc- 5, acc +5):
                    conv_c += 1
                else:
                    acc = new_accs
                    conv_c = 0
                if conv_c == conv :
                    print('.......CONVERGED........')
                    print(f"Training accuracy has converged to around {acc}% in the last {conv} epochs")
                    print(f'Trial {k}; highest accuracy reached was {accs}% for the values {best_w} as the Weights(W) and {best_b} as the Bias(b)')
                    break
                if epoch == 2500:
                    print("The perceptron didn't converge and had to be shutdown")
                    print(f"Training accuracy has converged to around {acc}% in the last {conv} epochs")
                    print(f'Trial {k}; highest accuracy reached was {accs}% for the values {best_w} as the Weights(W) and {best_b} as the Bias(b)')
                    break
                epoch += 1    
            if accs >= max_accs:
                max_accs = accs
                ultm_w = best_w
                ultm_b = best_b
            self.W = np.array(np.random.rand(1,self.num_of_weights))
            self.b = np.random.rand(1)

        print(f'Maximum accuracy reached through all trials and epochs is {max_accs}% for the values {ultm_w} as the weights(W) and {ultm_b} as the bias(b)')

    def accuracy(self, y_true, y_pred):
        acc = 0
        for i,j in zip(y_true,y_pred):
            if i == j:
                acc += 1
        return acc/len(y_true)  

    def step(self, y_pred):
        return 1 if y_pred >= 0 else 0
    def sigmoid(self, y_pred):
        return 1/(1 + exp(-y_pred))
#example
data = np.array(pd.read_csv('c:/users/kais/desktop/projects/projects/dt.csv'))
X = data[:,:2]
y = data[:,2]
perceptron = Perceptron(2)
pred = perceptron.fit_convergence(X, y_true = y, alpha = 0.001, conv = 10, trials  = 250)
