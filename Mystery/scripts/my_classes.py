import numpy as np
import keras

# class MysterySequencer(keras.utils.Sequence):
#     def __init__(self, x_set, y_set, nb_classes, batch_size):
#         self.x, self.y = x_set, y_set
#         self.nb_classes = nb_classes
#         self.batch_size = batch_size
#     def __len__(self) -> int:
#         return int(np.ceil(len(self.x) / float(self.batch_size)))
#     def __getitem__(self, idx) -> tuple:
#         batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
#         batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
#         return batch_x, keras.utils.to_categorical(batch_y, self.nb_classes)

class MysterySequencer(keras.utils.Sequence):
    def __init__(self, x_set, y_set, nb_classes, batch_size, is_categorical):
        self.x, self.y = x_set, y_set
        self.nb_classes = nb_classes
        self.batch_size = batch_size
        self.is_categorical = is_categorical
    def __len__(self) -> int:
        return int(np.ceil(len(self.x) / float(self.batch_size)))
    def __getitem__(self, idx) -> tuple:
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        if self.is_categorical:
            batch_y = keras.utils.to_categorical(batch_y, self.nb_classes)
        return batch_x, batch_y

def sample(array):
    print("data size = " + str(array.size))
    print("data shape = " + str(array[0].size))
    if array.size > 10:
        array_size=10
    else:
        array_size=array.size
    if array[0].size > 30:
        array_size2=30
    else:
        array_size2=array[0].size
    for i in range((array_size)):
        print("data nb"+str(i))
        for j in range((array_size2)):
            print(str(array[i][j]) + "|",end='')
        print("")


activations_shortcut = {'sigmoid' : 'Si',
                        'softsign' : 'Ss',
                        'softplus' : 'Sp',
                        'elu' : 'E',
                        'selu' : 'Se',
                        'softmax' : 'Sm',
                        'tanh' : 'T',
                        'relu': 'R',
                        'hard_sigmoid' :'Hs',
                        'linear': 'L'}

loss_shortcut = {'categorical_crossentropy' : 'cc',
                 'kullback_leibler_divergence' : 'kld',
                 'mean_squared_error' : 'mse',
                 'mean_absolute_error' : 'mae',#
                 'squared_hinge' : 'sh',#
                 'hinge' : 'h',##
                 'logcosh' : 'lc',##
                 'categorical_hinge' : 'ch',
                 'cosine_proximity' : 'cp',
                 'poisson' : 'po' }#

optimizer_shortcut = {'sgd' : 'sgd',
                      'rmsprop' : 'rm',
                      'Adagrad' : 'adag',#
                      'Adadelta' : 'adad',#
                      'Adam' :'adam',#
                      'Adamax' : 'adax',#
                      'Nadam' : 'nada',#
                      'TFOptimizer' : 'tf' }#


def check_input(input):
    for item in input:
        j=0;
        while(1):
            gotOne = False
            for i in range(5):
                if item[i+j] == 1:
                    if(gotOne):
                        print("there is two 1")
                    else:
                        gotOne=True
            j=j+15
            if(gotOne == False):
                print("The is no one")
            if(j+5 < item.size):
                break
    return 1