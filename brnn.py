from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.datasets import imdb

import types
import theano
import theano.tensor as T

from keras.layers.recurrent import Recurrent, GRU

'''
    Train a Bidirectional-LSTM on the IMDB sentiment classification task.
    Code borrowed from Keras/examples
    The dataset is actually too small for LSTM to be of any advantage
    compared to simpler, much faster methods such as TF-IDF+LogReg.
    Notes:
    - RNNs are tricky. Choice of batch size is important,
    choice of loss and optimizer is critical, etc.
    Some configurations won't converge.
    - LSTM loss decrease patterns during training can be quite different
    from what you see with CNNs/MLPs/etc.
    GPU command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python imdb_lstm.py
'''
def _get_reversed_input(self, train=False):
    if hasattr(self, 'previous'):
        X = self.previous.get_output(train=train)
    else:
        X = self.input
    return X[::-1]

class Bidirectional(Recurrent):
    def __init__(self, forward=None, backward=None, return_sequences=False,
                 forward_conf=None, backward_conf=None):
        assert forward is not None or forward_conf is not None, "Must provide a forward RNN or a forward configuration"
        assert backward is not None or backward_conf is not None, "Must provide a backward RNN or a backward configuration"
        super(Bidirectional, self).__init__()
        if forward is not None:
            self.forward = forward
        else:
            # Must import inside the function, because in order to support loading
            # we must import this module inside layer_utils... ugly
            from keras.utils.layer_utils import container_from_config
            self.forward = container_from_config(forward_conf)
        if backward is not None:
            self.backward = backward
        else:
            from keras.utils.layer_utils import container_from_config
            self.backward = container_from_config(backward_conf)
        self.return_sequences = return_sequences
        self.output_dim = self.forward.output_dim + self.backward.output_dim

        if not (self.return_sequences == self.forward.return_sequences == self.backward.return_sequences):
            raise ValueError("Make sure 'return_sequences' is equal for self,"
                             " forward and backward.")

    def build(self):
        self.input = T.tensor3()
        self.forward.input = self.input
        self.backward.input = self.input
        self.forward.build()
        self.backward.build()
        self.params = self.forward.params + self.backward.params

    def set_previous(self, layer, connection_map={}):
        assert self.nb_input == layer.nb_output == 1, "Cannot connect layers: input count and output count should be 1."
        if hasattr(self, 'input_ndim'):
            assert self.input_ndim == len(layer.output_shape), "Incompatible shapes: layer expected input with ndim=" +\
                str(self.input_ndim) + " but previous layer has output_shape " + str(layer.output_shape)
        self.forward.set_previous(layer, connection_map)
        self.backward.set_previous(layer, connection_map)
        self.backward.get_input = types.MethodType(_get_reversed_input, self.backward)
        self.previous = layer
        self.build()

    @property
    def output_shape(self):
        input_shape = self.input_shape
        output_dim = self.output_dim
        if self.return_sequences:
            return (input_shape[0], input_shape[1], output_dim)
        else:
            return (input_shape[0], output_dim)

    def get_output(self, train=False):
        Xf = self.forward.get_output(train)
        Xb = self.backward.get_output(train)
        Xb = Xb[::-1]
        return T.concatenate([Xf, Xb], axis=-1)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'forward_conf': self.forward.get_config(),
                'backward_conf': self.backward.get_config(),
                'return_sequences': self.return_sequences}


wordvec_model_path = ""

max_features = 20000
maxlen = 100  # cut texts after this number of words
batch_size = 32

print("Loading data...")
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features, test_split=0.2)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print("Pad sequences (samples x time)")
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

lstm = LSTM(output_dim=64)
gru = GRU(output_dim=64)  # original examples was 128, we divide by 2 because results will be concatenated
brnn = Bidirectional(forward=lstm, backward=gru)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen))
model.add(brnn)  # try using another Bidirectional RNN inside the Bidirectional RNN. Inception meets callback hell.
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")

print("Train...")
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=4, validation_data=(X_test, y_test), show_accuracy=True)
score, acc = model.evaluate(X_test, y_test, batch_size=batch_size, show_accuracy=True)
print('Test score:', score)
print('Test accuracy:', acc)