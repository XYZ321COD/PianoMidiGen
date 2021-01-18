from magenta.pipelines import pipeline
from magenta.pipelines import statistics
import IPython
from collections.abc import Iterable
import numpy as np
from ..project_utils.logger_ import create_logger

logger = create_logger(__name__)


class PianoRollsToIntoNoteAndTimeSteps(pipeline.Pipeline):

    def __init__(self, name="Transform piano rolls into note and time steps"):
        """Transform pianoRoll into NoteAndTimeStep format

        :param: (string) name: Pipeline name.
        """
        super(PianoRollsToIntoNoteAndTimeSteps, self).__init__(
            input_type=Iterable,
            output_type=Iterable,
            name=name)

        self.stat1 = statistics.Counter('how_many_notes')
        self.stats = [self.stat1]

    def transform(self, dict_time_notes, seq_len=50, fp=30):
        """Transforming midi format into pianoRoll format

        :param: (dict) dict_time_notes: PianoRolls format of midi files.
        :param: (float) seq_len: length of sequences
        :param: (float) fp: frames per second
        :return: list of all midi files but in NoteAndTimeSteps Format
        :rtype: list
        """
        list_of_dict_keys_time = []

        for key in dict_time_notes:
            dict_keys_time = {}
            piano_roll = dict_time_notes[key]
            # I use np.unique, could use set instead
            times = np.unique(np.where(piano_roll > 0)[1])
            # Returns indices of elements < 0  of 2D array example: (array([0, 0, 0], dtype=int64), array([0, 1, 2], dtype=int64))
            index = np.where(piano_roll > 0)
            for time in times:
                notes = index[0][np.where(index[1] == time)]
                dict_keys_time[time] = notes
                self.stat1.increment(len(notes))
            list_of_dict_keys_time.append(dict_keys_time)
            logger.info("Create {} notes and the song length {} mins".format(
                self.stat1, len(times)//(60*fp)))
        return list_of_dict_keys_time

    def get_stats(self):
        """Returns stats about pipeline
        """
        return self.stats
