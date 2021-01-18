import numpy as np
import pretty_midi
from magenta.pipelines import pipeline
from magenta.pipelines import statistics
from utils.project_utils.logger_ import create_logger, mkdir_p
import utils.pipelines as df
from collections.abc import Iterable


logger = create_logger(__name__)


class ModelOutPutIntoMidiFile(pipeline.Pipeline):
    """Transform model output into midi file format

    :param: (string) name: Pipeline name.
    """

    def __init__(self, name="Transform model output into midi file format"):
        super(ModelOutPutIntoMidiFile, self).__init__(
            input_type=Iterable,
            output_type=Iterable,
            name=name)

        self.stat1 = statistics.Counter('how_many_notes')
        self.stats = [self.stat1]

    def transform(self, note_tokenizer, generate, midi_file_name="result.mid", start_index=49, fs=5, max_generated=10000):
        '''Convert a model output into piano roll and after that into midi format

        :param: (NoteTokenizer) note_tokenizer :   instance of class holdin info about training set
        :param: (np.darray) generate :  generated data by model
        :param: (string)  midi_file_name : path to save music to
        :param: (int) start_index:  size of the window
        :param: (int) fs: frame per second
        :param: (int) max_generated: defining the length of generated song
        :return: midi_file
        :rtype: MIDI
        '''
        note_string = [note_tokenizer.index_to_notes[ind_note]
                       for ind_note in generate]
        array_piano_roll = np.zeros(
            (128, max_generated + fs), dtype=np.int16)
        for index, note in enumerate(note_string[start_index:]):
            if note == 'empty':
                pass
            else:
                splitted_note = note.split(',')
                for j in splitted_note:
                    array_piano_roll[int(j), index] = 1
        generate_to_midi = df.pipeline5.NotesAndTimesStepIntoMidi().transform(
            array_piano_roll, fs=fs)
        print("Tempo {}".format(generate_to_midi.estimate_tempo()))
        for note in generate_to_midi.instruments[0].notes:
            note.velocity = 70
        generate_to_midi.write(midi_file_name)

    def get_stats(self):
        """Returns stats about pipeline
        """
        return self.stats
