import os
import numpy as np
import music21 as m21
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Function to convert MIDI files to note sequences
def midi_to_notes(midi_path):
    try:
        print(f"Parsing MIDI file: {midi_path}")
        midi = m21.converter.parse(midi_path)
        notes = []
        for element in midi.flat.notes:
            if isinstance(element, m21.note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, m21.chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
        return notes
    except Exception as e:
        print(f"Error parsing {midi_path}: {e}")
        return []

# Function to recursively find all MIDI files in a directory
def find_midi_files(directory):
    midi_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.mid'):
                midi_files.append(os.path.join(root, file))
    return midi_files

midi_directory = r"C:\Users\fathi\Desktop\codealpha\mus_gen_2\CodeAlpha_Project2_-Music_Generation_with_AI\midi_file"
all_notes = []

midi_files = find_midi_files(midi_directory)
print(f"Found {len(midi_files)} MIDI files.")

for midi_file_path in midi_files:
    notes = midi_to_notes(midi_file_path)
    if notes:  # Only extend if notes were successfully extracted
        all_notes.extend(notes)

print(f"Total notes extracted: {len(all_notes)}")

if not all_notes:
    raise ValueError("No notes extracted from the MIDI files.")

# Encode the notes
note_names = sorted(set(all_notes))
encoder = LabelEncoder()
encoder.fit(note_names)
encoded_notes = encoder.transform(all_notes)

# Create input-output pairs
sequence_length = 100
network_input = []
network_output = []

for i in range(len(encoded_notes) - sequence_length):
    input_sequence = encoded_notes[i:i + sequence_length]
    output_sequence = encoded_notes[i + sequence_length]
    network_input.append(input_sequence)
    network_output.append(output_sequence)

network_input = np.reshape(network_input, (len(network_input), sequence_length, 1))
network_output = to_categorical(network_output, num_classes=len(note_names))

# Build and train the RNN model
model = Sequential()
model.add(LSTM(256, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(256))
model.add(Dense(len(note_names), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(network_input, network_output, epochs=50, batch_size=64)

# Function to generate music
def generate_music(model, network_input, encoder, num_generate=500):
    start_index = np.random.randint(0, len(network_input) - 1)
    pattern = network_input[start_index]

    generated_notes = []
    for _ in range(num_generate):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction = model.predict(prediction_input, verbose=0)
        index = np.argmax(prediction)
        result = encoder.inverse_transform([index])[0]
        generated_notes.append(result)
        pattern = np.append(pattern[1:], index)
        pattern = pattern[-len(network_input[0]):]  # keep the pattern length constant

    return generated_notes

# Generate new music
generated_notes = generate_music(model, network_input, encoder)
print("Generated notes:", generated_notes)

# Function to convert generated notes to MIDI
def notes_to_midi(notes, output_path='generated_music.mid'):
    output_notes = []
    offset = 0

    for note in notes:
        if ('.' in note) or note.isdigit():
            chord_notes = [m21.note.Note(int(n)) for n in note.split('.')]
            chord = m21.chord.Chord(chord_notes)
            chord.offset = offset
            output_notes.append(chord)
        else:
            note = m21.note.Note(note)
            note.offset = offset
            output_notes.append(note)
        offset += 0.5

    midi_stream = m21.stream.Stream(output_notes)
    midi_stream.write('midi', fp=output_path)

# Convert and save the generated notes to a MIDI file
output_midi_path = 'C:\\Users\\ASUS\\Desktop\\generated_music.mid'
notes_to_midi(generated_notes, output_path=output_midi_path)
print(f"Generated MIDI file saved to: {output_midi_path}")

