import numpy as np
import pretty_midi
from magenta.pipelines import pipeline
from magenta.pipelines import statistics
from utils.project_utils.logger_ import create_logger
from collections.abc import Iterable


logger = create_logger(__name__)


class NotesAndTimesStepIntoMidi(pipeline.Pipeline):
    """Transformat NotesAndTimesSTep format into Midi

    :param: (string) name: Pipeline name.
    """

    def __init__(self, name="Transformat NotesAndTimesSTep format into Midi"):
        super(NotesAndTimesStepIntoMidi, self).__init__(
            input_type=Iterable,
            output_type=Iterable,
            name=name)

        self.stat1 = statistics.Counter('how_many_notes')
        self.stats = [self.stat1]

    def transform(self, piano_roll, fs=5, program=0):
        '''Convert a Piano Roll array into a PrettyMidi object
        with a single instrument

        :param: (np.ndarray) piano_roll : Piano roll of piano instrument
        :param: (int) fs : Sampling frequency of the columns
        :param: (int) program: The program number of the instrument
        :return: A pretty_midi.PrettyMIDI class instance describing the piano roll.
        :rtype: pretty_midi.PrettyMIDI
        '''
        notes, frames = piano_roll.shape
        pm = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=program)

        piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

        velocity_changes = np.nonzero(np.diff(piano_roll).T)

        prev_velocities = np.zeros(notes, dtype=int)
        note_on_time = np.zeros(notes)

        for time, note in zip(*velocity_changes):
            velocity = piano_roll[note, time + 1]
            time = time / fs
            if velocity > 0:
                if prev_velocities[note] == 0:
                    note_on_time[note] = time
                    prev_velocities[note] = velocity
            else:
                pm_note = pretty_midi.Note(
                    velocity=prev_velocities[note],
                    pitch=note,
                    start=note_on_time[note],
                    end=time)
                instrument.notes.append(pm_note)
                prev_velocities[note] = 0
        pm.instruments.append(instrument)
        return pm

    def get_stats(self):
        """Returns stats about pipeline
        """
        return self.stats
