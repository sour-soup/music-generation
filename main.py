from utils import midi_to_notes, create_midi

if __name__ == '__main__':
    input_file = 'data/sample.mid'
    output_file = 'data/sample_output.mid'
    notes = midi_to_notes(input_file, quantization=32)
    print(notes)
    create_midi(notes, output_file, bpm=240, quantization=32)

