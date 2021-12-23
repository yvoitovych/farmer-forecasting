import numpy as np
import torch
torch.mean(torch.abs((target - output) / target))



class MAPE:
    NAME = 'MAPE'

    def __init__(self):
        self.e = 10e-8
        self.clear()


    def clear(self):
        self.confusion_matrix = np.zeros((2, 2))
        # self.counter = 0

    def add(self, y_hat, y):
        print(y, y_hat)
        TP = (y[y>0.5]&y_hat[y_hat==1]).sum()
        FP = (y[y<=0.5]&y_hat[y_hat==1]).sum()
        FN = (y[y>0.5]&y_hat[y_hat==0]).sum()
        TN = (y[y<=0.5]&y_hat[y_hat==0]).sum()
        self.confusion_matrix[0, 0] += TP
        self.confusion_matrix[1, 0] += FP
        self.confusion_matrix[0, 1] += FN
        self.confusion_matrix[1, 1] += TN

    def return_value(self):
        value = self.confusion_matrix[0,0]/(self.confusion_matrix[0].sum() + self.e)
        print(f'{self.NAME}: {value};')

        return value
