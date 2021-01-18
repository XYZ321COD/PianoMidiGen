from magenta.pipelines import pipeline
from magenta.pipelines import statistics
from tqdm import tnrange, notebook, tqdm
import IPython
import pretty_midi
from collections.abc import Iterable
import numpy as np
from ..project_utils.logger_ import create_logger

logger = create_logger(__name__)


class MidiFilesToPianoRoll(pipeline.Pipeline):

    def __init__(self, name="Transform midi files to piano roll format"):
        """Extract the notes for desired format - binary 2D numpy.array in (notes, time)

        :param: (string) name: Pipeline name.
        """
        super(MidiFilesToPianoRoll, self).__init__(
            input_type=str,
            output_type=Iterable,
            name=name)
        self.stat1 = statistics.Counter('how_many_midi_file_processed')
        self.stats = [self.stat1]

    def transform(self, list_all_midi, fs):
        """Transforming midi format into pianoRoll format

        :param: (list) list_all_midi: list of all midi files
        :param: (float) fs:Sampling frequency of the columns, i.e. each column is spaced apart by 1./fs seconds.
        :return: dict of format piano_roll - typical way of processing midi files
        :rtype: dict
        """
        dict_time_notes = {}
        process_tqdm_midi = tqdm(range(0, len(list_all_midi)))
        for i in process_tqdm_midi:
            midi_file_name = list_all_midi[i]
            logger.info("Processing {}".format(midi_file_name))
            try:
                midi_pretty_format = pretty_midi.PrettyMIDI(midi_file_name)
                piano_midi = midi_pretty_format.instruments[0]
                piano_roll = piano_midi.get_piano_roll(fs=fs)
                dict_time_notes[i] = piano_roll
                logger.info(" Sucessfully Processed {}".format(midi_file_name))
                self.stat1.increment(1)
            except Exception as e:
                logger.error(
                    "Broken file : {} with error trace {}".format(midi_file_name, e))
        return dict_time_notes

    def get_stats(self):
        """Returns stats about pipeline
        """
        return self.stats
