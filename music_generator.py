from music21 import converter, instrument, note, chord, stream
import os
import numpy as np
import tensorflow as tf
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
from keras.callbacks import ModelCheckpoint

def extract_notes(file_path):
    """
    Extracts notes and chords from a MIDI file.
    """
    midi = converter.parse(file_path)
    notes_to_parse = None
    try:  # file has instrument parts
        s2 = instrument.partitionByInstrument(midi)
        notes_to_parse = s2.parts[0].recurse()
    except:  # file has notes in a flat structure
        notes_to_parse = midi.flat.notes
    notes = []
    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))
    return notes

def prepare_data(lmd_path):
    all_notes = []
    for subdir, dirs, files in os.walk(lmd_path):
        for filename in files:
            if filename.endswith(".mid"):
                file_path = os.path.join(subdir, filename)
                notes = extract_notes(file_path)
                all_notes.extend(notes)

    unique_notes = sorted(set(all_notes))
    return all_notes, unique_notes

def prepare_sequences(all_notes, unique_notes, sequence_length=100):
    """
    Prepare the sequences used by the Neural Network
    """
    note_to_int = dict((note, number) for number, note in enumerate(unique_notes))

    network_input = []
    network_output = []

    for i in range(0, len(all_notes) - sequence_length, 1):
        sequence_in = all_notes[i:i + sequence_length]
        sequence_out = all_notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # Reshape for LSTM layer
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    network_input = network_input / float(len(unique_notes))  # Normalize input

    network_output = np_utils.to_categorical(network_output)

    return network_input, network_output

def create_model(unique_notes, sequence_length=100):
    """
    Create the structure of the neural network
    """
    model = Sequential()
    model.add(LSTM(512, input_shape=(sequence_length, 1), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(len(unique_notes)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model

def train_model(model, network_input, network_output):
    """
    Train the neural network
    """
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, epochs=200, batch_size=64, callbacks=callbacks_list)

def generate_music(model, network_input, unique_notes, sequence_length=100, num_notes=500):
    """
    Generate notes using the trained model
    """
    start = np.random.randint(0, len(network_input) - 1)
    int_to_note = dict((number, note) for number, note in enumerate(unique_notes))

    pattern = network_input[start]
    generated_notes = []

    for note_index in range(num_notes):
        prediction_input = np.reshape(pattern, (1, sequence_length, 1))
        prediction_input = prediction_input / float(len(unique_notes))

        prediction = model.predict(prediction_input, verbose=0)
        index = np.argmax(prediction)
        result = int_to_note[index]
        generated_notes.append(result)

        pattern = np.append(pattern, index)
        pattern = pattern[1:len(pattern)]

    return generated_notes

def create_midi(generated_notes, output_file='output.mid'):
    """
    Convert the output from the prediction to notes and create a MIDI file from the notes
    """
    offset = 0
    output_notes = []

    for pattern in generated_notes:
        # if the pattern is a chord
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
        # if the pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=output_file)

# Set the path to the Lakh MIDI Dataset
lmd_path = "path/to/lmd"

# Prepare the data
all_notes, unique_notes = prepare_data(lmd_path)

# Create the model
model = create_model(unique_notes)

# Prepare sequences for training
network_input, network_output = prepare_sequences(all_notes, unique_notes)

# Train the model
train_model(model, network_input, network_output)

# Generate music
generated_notes = generate_music(model, network_input, unique_notes)

# Create MIDI file
create_midi(generated_notes)
