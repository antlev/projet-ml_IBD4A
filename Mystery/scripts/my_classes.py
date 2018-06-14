import numpy as np
import keras

class MysterySequencer(keras.utils.Sequence):
    def __init__(self, x_set, y_set, nb_classes, batch_size):
        self.x, self.y = x_set, y_set
        self.nb_classes = nb_classes
        self.batch_size = batch_size
    def __len__(self) -> int:
        return int(np.ceil(len(self.x) / float(self.batch_size)))
    def __getitem__(self, idx) -> tuple:
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, keras.utils.to_categorical(batch_y, self.nb_classes)

