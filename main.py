""" ** midi 파일 데이터를 Neural Network 에 입력하여 학습시키는 모듈 ** """

import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

def train_network():
    """ ** 음악을 생성하기 위한 신경망 훈련 ** """

    notes = get_notes()

    # 음높이 이름(계이름)을 입력 받음
    # pitch name => 알파벳과 숫자를 조합하여 소리의 pitch(음높이) 표시법
    n_vocab = len(set(notes))

    network_input, network_output = prepare_sequences(notes, n_vocab)

    model = create_network(network_input, n_vocab)

    train(model, network_input, network_output)

def get_notes():
    """ ** ./sample 디렉토리의 midi 파일에서 모든 노트(음표)와 코드를 가져옴 ** """

    notes = []

    for file in glob.glob("sample/Violet_Evergarden/*.mid"):
        midi = converter.parse(file)

        print("Parsing %s" % file)

        notes_to_parse = None

        try: # 파일이 instrument parts(악기 부분)를 가지고 있을 때
            parts = instrument.partitionByInstrument(midi)
            notes_to_parse = parts.parts[0].recurse()
        except: # 파일이 플랫 노트(반음 내림)를 가지고 있을 때
            notes_to_parse = midi.flat.notes            

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
                
    # 모델을 훈련시키는 데 사용된 노트 load
    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes

def prepare_sequences(notes, n_vocab):
    """ ** 신경망에서 사용할 시퀀스 준비 ** """

    sequence_length = 100

    # 음높이 이름(계이름)을 얻음
    pitchnames = sorted(set(item for item in notes))

    # 음높이(pitch)를 정수(integers)로 mapping 하는 dictionary 구축
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # 입력(input)과 출력(output) 시퀀스 생성
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]

        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # 입력(input)을 LSTM 레이어와 호환되는 형식으로 재구성
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))

    # 입력(input) normalize
    network_input = network_input / float(n_vocab)

    network_output = np_utils.to_categorical(network_output)

    return (network_input, network_output)

def create_network(network_input, n_vocab):
    """ ** 신경망의 구조 생성(구축) ** """

    model = Sequential()
    model.add(LSTM(
        512,
        input_shape = (network_input.shape[1], network_input.shape[2]),
        recurrent_dropout = 0.3,
        return_sequences = True
    ))
    model.add(LSTM(512, return_sequences = True, recurrent_dropout = 0.3,))
    model.add(LSTM(512))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop')

    return model

def train(model, network_input, network_output):
    """ ** 신경망 훈련 (training) ** """

    filepath = "logs/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor = 'loss',
        verbose = 0,
        save_best_only = True,
        mode = 'min'
    )

    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, epochs = 200, batch_size = 128, callbacks = callbacks_list)

if __name__ == '__main__':
    train_network()