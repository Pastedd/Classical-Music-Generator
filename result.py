""" ** 학습된 신경망을 통해 midi 파일에 대한 노트(음표) 생성하는 모듈 ** """

import pickle
import numpy as np
from music21 import instrument, note, stream, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import BatchNormalization as BatchNorm
from keras.layers import Activation

def generate():
    """ ** midi 파일 생성 ** """

    # 모델을 훈련시키는 데 사용된 노트 load
    with open('data/notes', 'rb') as filepath:
        notes = pickle.load(filepath)

    # 음높이 이름(계이름)을 입력 받음
    pitchnames = sorted(set(item for item in notes))
    n_vocab = len(set(notes))

    network_input, normalized_input = prepare_sequences(notes, pitchnames, n_vocab)
    model = create_network(normalized_input, n_vocab)
    prediction_output = generate_notes(model, network_input, pitchnames, n_vocab)
    create_midi(prediction_output)

def prepare_sequences(notes, pitchnames, n_vocab):
    """ ** 신경망에서 사용할 시퀀스 준비 ** """

    # 노트와 정수 사이의 mapping
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    sequence_length = 100

    network_input = []
    output = []

    # 입력(input)과 출력(output) 시퀀스 생성
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]

        network_input.append([note_to_int[char] for char in sequence_in])
        output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # 입력(input)을 LSTM 레이어와 호환되는 형식으로 재구성
    normalized_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    
    # 입력(input) normalize
    normalized_input = normalized_input / float(n_vocab)

    return (network_input, normalized_input)

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

    # 가중치를 각 노드에 load
    # ex) model.load_weights('weights.hdf5') <= 가중치.hdf5 파일 입력
    model.load_weights('logs/weights-improvement-28-6.1792-bigger.hdf5')

    return model

def generate_notes(model, network_input, pitchnames, n_vocab):
    """ ** 일련의 노트를 기반으로 신경망에서 새로운 노트 생성 ** """

    # 예측된 시작점을 통해 입력에서 랜덤으로 시퀀스를 선택
    start = np.random.randint(0, len(network_input)-1)

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    pattern = network_input[start]
    prediction_output = []

    # 500 개의 노트를 생성
    for note_index in range(500):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose = 0)

        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return prediction_output

def create_midi(prediction_output):
    """ ** 예측된 output 을 노트로 변환하고 midi 파일 생성 ** """

    offset = 0
    output_notes = []

    # 모델에 의해 생성된 값을 기반으로 노트 및 코드의 객체를 생성
    for pattern in prediction_output:
        # 코드(chord) 패턴 생성
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)

        # 노트 패턴 생성
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # 중복된 노트가 쌓이지 않도록 각 반복 간격을 0.5 씩 띄우기(증가)
        offset += 0.5

    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp = 'logs/new_notes.mid')

if __name__ == '__main__':
    generate()