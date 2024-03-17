from keras.layers import Input, Dense, ActivityRegularization
from keras.models import Sequential

from archimedes import Archimedes
from utils import CLASSES

EMBEDDING_DIM = 1024


def get_encoder(in_dim):
    print('[INFO] Building Encoder')
    model = Sequential(name='encoder')
    model.add(Input(shape=(in_dim,)))
    model.add(Dense(2048, activation='selu'))
    model.add(Dense(EMBEDDING_DIM, activation='selu'))
    model.add(ActivityRegularization(l1=1e-3))
    return model


def get_decoder():
    print('[INFO] Building Decoder')
    model = Sequential(name='decoder')
    model.add(Input(shape=(EMBEDDING_DIM,)))
    model.add(Dense(2048, activation='selu'))
    model.add(Dense(len(CLASSES), activation='softmax'))
    return model


def build_ssae(in_dim):
    encoder = get_encoder(in_dim)
    decoder = get_decoder()
    print('[INFO] Building Stacked Sparse AutoEncoder')
    model = Sequential([encoder, decoder])
    print('[INFO] Compiling Model Using Archimedes Optimizer')
    model.compile(Archimedes(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    build_ssae(4048)
