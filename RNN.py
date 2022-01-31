import csv
import numpy as np
import emoji
from emo_utils import *
from matplotlib import pyplot as plt

class data:
    def __init__(self, is_train):
        self.is_train = is_train
        print('init')

    def dataloader(self):
        path_1 = "./data_train_test/" # you have to change it according to your path
        path_2 = "../glove.6B/"
        if self.is_train == True:
            X, Y = read_csv(path_1 + 'train_emoji.csv') # x에 대문자 포함해서 문장 단위가 들어옴
        elif self.is_train == False:
            X, Y = read_csv(path_1 + 'test_emoji.csv')
        Y_onehot = []
        X_split = []
        for i in range(len(X)):
            X_split.append(X[i].lower().split())
            onehot = np.zeros((1, 5))
            onehot[0][Y[i]] = 1
            Y_onehot.append(onehot)

        glove_word_to_index, glove_index_to_word, glove_word_to_array = read_glove_vecs(path_2 + "glove.6B.50d.txt")
        X_input = []

        for i in range(len(X_split)):
            word_vector = []
            for word in X_split[i]:
                off = np.ones((1, 50))
                for i in range(50):
                    off[0][i] = glove_word_to_array[word][i]
                word_vector.append(off)
            X_input.append(word_vector)

        X_input = np.array(X_input) # 1*50 vector
        Y_onehot = np.array(Y_onehot) # 1*5 one hot vector
        
        return X_input, Y_onehot, X_split

class function:
    def Cross_Entropy_Loss(self, softmax_matrix, label_matrix):
        delta = 1e-7 
        return -np.sum(label_matrix*np.log(softmax_matrix+delta))

    def softmax(self, x):
        s = np.exp(x)
        total = np.sum(s, axis=0).reshape(-1,1)
        return s/total

    def tanh(self, x):
        s = (1 - np.exp(-1*x))/(1 + np.exp(-1*x))
        return s

    def differ_softmax_cross(self, softmax_matrix, label_matrix):
        x = (softmax_matrix - label_matrix)
        return x

    def differ_tanh(self, x):
        output = 1 - self.tanh(x)*self.tanh(x)
        return output

class RNN_cell(function):
    def __init__(self, hidden_size, input_size):
        np.random.seed(0)
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.Whh = np.random.randn(hidden_size, hidden_size) / np.sqrt(hidden_size)
        self.Wxh = np.random.randn(hidden_size, input_size) / np.sqrt(hidden_size)
        self.b = np.random.randn(hidden_size, 1) / np.sqrt(hidden_size)
        self.moment_Whh_1 = 0
        self.moment_Whh_2 = 0
        self.moment_Wxh_1 = 0
        self.moment_Wxh_2 = 0
        self.moment_b_1 = 0
        self.moment_b_2 = 0
        self.rho = 0.9
        self.decay_rate = 0.9

    def rnn_cell_forward(self, h_input, x_input):
        self.h_input = h_input
        self.x_input = x_input
        self.h_output = super().tanh(np.dot(self.Whh, h_input) + np.dot(self.Wxh, x_input) + self.b)
        self.diff_h_output = super().differ_tanh(np.dot(self.Whh, h_input) + np.dot(self.Wxh, x_input) + self.b)

    def rnn_cell_backward(self, dh_next): # diff_h_output
        dtanhx = np.multiply(dh_next, self.diff_h_output) # 20*1, 20*1 = 20*1
        self.dWhh = np.dot(dtanhx, self.h_input.T) # 20*1, 1*20 = 20*20
        self.dWxh = np.dot(dtanhx, self.x_input.T) # 20*1, 1*50 = 20*50 or 20*1, 1*20 = 20*20
        self.db = dtanhx

        self.dh_prev = np.dot(self.Whh, dtanhx)
        if self.input_size == self.hidden_size:
            self.dx_prev = np.dot(self.Wxh, dtanhx)

        self.dh_next = dh_next

    def SGD(self, learning_rate):
        self.Whh -= learning_rate * self.dWhh
        self.Wxh -= learning_rate * self.dWxh
        self.b -= learning_rate * self.db

    def ADAM(self, learning_rate):
        # self.Whh
        self.moment_Whh_1 = self.rho * self.moment_Whh_1 + (1 - self.rho) * self.dWhh
        self.moment_Whh_2 = self.decay_rate * self.moment_Whh_2 + (1 - self.decay_rate) * self.dWhh * self.dWhh
        self.Whh -= learning_rate * self.moment_Whh_1 / (np.sqrt(self.moment_Whh_2) + 1e-7)

        # self.Wxh
        self.moment_Wxh_1 = self.rho * self.moment_Wxh_1 + (1 - self.rho) * self.dWxh
        self.moment_Wxh_2 = self.decay_rate * self.moment_Wxh_2 + (1 - self.decay_rate) * self.dWxh * self.dWxh
        self.Wxh -= learning_rate * self.moment_Wxh_1 / (np.sqrt(self.moment_Wxh_2) + 1e-7)

        # self.b
        self.moment_b_1 = self.rho * self.moment_b_1 + (1 - self.rho) * self.db
        self.moment_b_2 = self.decay_rate * self.moment_b_2 + (1 - self.decay_rate) * self.db * self.db
        self.b -= learning_rate * self.moment_b_1 / (np.sqrt(self.moment_b_2) + 1e-7)

class RNN_MODEL_2_layer(RNN_cell):
    def __init__(self, X_input, Y_input, X_test_input, Y_test_input, X_split, X_test_split, learning_rate, epoch, optimizer, dropout):
        np.random.seed(0)
        self.X_input = X_input
        self.Y_input = Y_input
        self.X_test_input = X_test_input
        self.Y_test_input = Y_test_input
        self.X_split = X_split
        self.X_test_split = X_test_split
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.optimizer = optimizer
        self.dropout = dropout
        self.RNN_1st_layer = [RNN_cell(hidden_size = 40, input_size = 50) for _ in range(10)]
        self.RNN_2nd_layer = [RNN_cell(hidden_size = 40, input_size = 40) for _ in range(10)]
        self.Why = np.random.randn(5, self.RNN_2nd_layer[0].hidden_size) / np.sqrt(self.RNN_2nd_layer[0].hidden_size)
        self.by = np.random.randn(5, 1) / np.sqrt(self.RNN_2nd_layer[0].hidden_size)
        self.moment_Why_1 = 0
        self.moment_Why_2 = 0
        self.moment_by_1 = 0 
        self.moment_by_2 = 0
        self.rho = 0.9
        self.decay_rate = 0.9
        self.h_before = np.zeros((self.RNN_2nd_layer[0].hidden_size, 1))
        self.loss, self.loss_list = [], []
        self.test_loss, self.test_loss_list = [], []
        self.train_accuracy_list, self.test_accuracy_list = [], []
        self.train_accuracy = 0
        self.test_accuracy = 0
        self.train_prediction, self.test_prediction = [], []
        self.answer_emoji = []

    def model_start(self):
        for epoch in range(self.epoch):
            # train
            for iteration in range(len(self.X_input)):
                x_inputs = self.X_input[iteration]
                y_inputs = self.Y_input[iteration]

                self.forward(x_inputs, y_inputs, is_train = True)
                self.backward(x_inputs)

            loss = round(sum(self.loss)/len(X_input), 2)
            train_accuracy = round(self.train_accuracy/len(X_input) * 100, 2)

            if epoch % 10 == 0:
                print("train... ", epoch, " epoch -> loss :", loss, " train accuracy : ", train_accuracy, "%")

            self.loss_list.append(loss)
            self.train_accuracy_list.append(train_accuracy)
            self.loss = [] 
            self.train_accuracy = 0

            # test
            for iteration in range(len(self.X_test_input)):
                x_test_inputs = self.X_test_input[iteration]
                y_test_inputs = self.Y_test_input[iteration]
                self.forward(x_test_inputs, y_test_inputs, is_train = False)

            loss = round(sum(self.test_loss)/len(X_test_input), 2)
            test_accuracy = round(self.test_accuracy/len(X_test_input) * 100, 2)

            if epoch % 10 == 0:
                print("test ... ", epoch, " epoch -> loss :", loss, " test accuracy : ", test_accuracy, "%")

            self.test_loss_list.append(round((sum(self.test_loss)/len(X_test_input)), 2))
            self.test_accuracy_list.append(test_accuracy)
            self.test_loss = []
            self.test_accuracy = 0

            # print emoji
            if epoch == self.epoch - 1:
                for i, items in enumerate(self.X_test_split):
                    temp = str(i) + ". "
                    prediction = self.test_prediction[i]
                    answer = self.answer_emoji[i]
                    for item in items:
                        temp += item + " "
                    temp += "->"
                    print(temp, " test : ", label_to_emoji(prediction)) # "answer : ", label_to_emoji(answer), 

            self.test_prediction = []
            self.train_prediction = []
            self.answer_emoji = []
    
            # train & test loss graph
            if epoch  == self.epoch - 1:
                plt.subplot(2, 2, 1)
                plt.title('RNN + SGD + 50d loss', fontsize=12)
                plt.plot(range(0, epoch+1), self.loss_list, 'b', label = 'train')
                plt.plot(range(0, epoch+1), self.test_loss_list, 'r', label = 'test')
                plt.ylabel('Cost')
                plt.xlabel('Epochs')
                plt.legend(loc='upper right')

                plt.subplot(2, 2, 2)
                plt.title('RNN + SGD + 50d accuracy', fontsize=12)
                plt.plot(range(0, epoch+1), self.train_accuracy_list, 'b', label = 'train')
                plt.plot(range(0, epoch+1), self.test_accuracy_list, 'r', label = 'test')
                plt.ylabel('accuracy (%)')
                plt.xlabel('Epochs')
                plt.legend(loc='upper right')
                plt.show()

    def forward(self, x_inputs, y_inputs, is_train):
        h_before_1 = self.h_before # 맨 처음엔 0
        h_before_2 = self.h_before
        y_input = y_inputs.reshape(5, 1)
        self.ht, self.diff_ht = [], []
        self.ht_2, self.diff_ht_2 = [], []
        
        for n, x_input in enumerate(x_inputs):
            x_input = x_input.reshape(50,1)
            self.RNN_1st_layer[n].rnn_cell_forward(h_before_1, x_input)
            h_before_1 = self.RNN_1st_layer[n].h_output

            # dropout layer
            if self.dropout == "yes":
                h_before_1 = self.dropout_layer(h_before_1, 0.5)
                self.ht.append(h_before_1)

            else:
                self.ht.append(h_before_1)

            self.RNN_2nd_layer[n].rnn_cell_forward(h_before_2, self.ht[n])
            h_before_2 = self.RNN_2nd_layer[n].h_output
            self.ht_2.append(h_before_2)
            
            if n == len(x_inputs) - 1:
                y_predic = np.dot(self.Why, self.ht_2[-1]) + self.by # 5*20, 20*1 = 5*1
                y_softmax = super().softmax(y_predic)
                loss = super().Cross_Entropy_Loss(y_softmax, y_input)
                self.diff_softmax_cross = super().differ_softmax_cross(y_softmax, y_input)

        if is_train == True:
            prediction = np.argmax(y_softmax)
            answer = np.argmax(y_input)
            if prediction == answer:
                self.train_accuracy += 1
            self.loss.append(loss)
            self.train_prediction.append(prediction)

        elif is_train == False:
            prediction = np.argmax(y_softmax)
            answer = np.argmax(y_input)
            if prediction == answer:
                self.test_accuracy += 1
            self.test_loss.append(loss)
            self.test_prediction.append(prediction)
            self.answer_emoji.append(answer)

    def dropout_layer(self, x, prob):
        u = np.random.rand(*x.shape) < prob
        x *= u
        return x

    def backward(self, x_inputs):
        self.last_dht_2 = np.dot(self.Why.T, self.diff_softmax_cross) # 2층에서 2층으로 전달하는 ht
        self.last_dht_1 = 0 # 1층에서 1층으로 전달하는 ht
        self.last_dx_1 = 0 # 2층에서 1층으로 내려가는 x
        for n in reversed(range(0, len(x_inputs))):
            self.RNN_2nd_layer[n].rnn_cell_backward(self.last_dht_2) # self.diff_ht_2[n]
            self.last_dht_2 = self.RNN_2nd_layer[n].dh_prev
            self.last_dx_1 = self.RNN_2nd_layer[n].dx_prev
            self.last_dht_1 += self.last_dx_1
            self.RNN_1st_layer[n].rnn_cell_backward(self.last_dht_1) # self.diff_ht[n]
            self.last_dht_1 = self.RNN_1st_layer[n].dh_prev

        for m in range(0, len(x_inputs)):
            if self.optimizer == "SGD":
                self.RNN_1st_layer[m].SGD(self.learning_rate)
                self.RNN_2nd_layer[m].SGD(self.learning_rate)
            else:
                self.RNN_1st_layer[m].ADAM(self.learning_rate)
                self.RNN_2nd_layer[m].ADAM(self.learning_rate)

        self.dWhy = np.dot(self.diff_softmax_cross, self.ht_2[-1].T)
        self.dby = self.diff_softmax_cross

        if self.optimizer == "SGD":
            self.Why -= self.learning_rate * self.dWhy
            self.by -= self.learning_rate * self.dby

        else:
            # self.Why
            self.moment_Why_1 = self.rho * self.moment_Why_1 + (1 - self.rho) * self.dWhy
            self.moment_Why_2 = self.decay_rate * self.moment_Why_2 + (1 - self.decay_rate) * self.dWhy * self.dWhy
            self.Why -= self.learning_rate * self.moment_Why_1 / (np.sqrt(self.moment_Why_2) + 1e-7)

            # self.by
            self.moment_by_1 = self.rho * self.moment_by_1 + (1 - self.rho) * self.dby
            self.moment_by_2 = self.decay_rate * self.moment_by_2 + (1 - self.decay_rate) * self.dby * self.dby
            self.by -= self.learning_rate * self.moment_by_1 / (np.sqrt(self.moment_by_2) + 1e-7)
        
if __name__ == "__main__":
    _data = data(is_train = True)
    _test_data = data(is_train = False)
    X_input, Y_input, X_split = _data.dataloader()
    X_test_input, Y_test_input, X_test_split = _test_data.dataloader()
    rnn = RNN_MODEL_2_layer(X_input, Y_input, X_test_input, Y_test_input, X_split, X_test_split, 0.001, 500, optimizer = "SGD", dropout = "no") # without dropout 0.03, 400, 5, 50 / with dropout 
    rnn.model_start()
