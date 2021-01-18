from magenta.pipelines import pipeline
from magenta.pipelines import statistics
from collections.abc import Iterable
from random import shuffle, seed
import IPython
import glob
from ..project_utils.logger_ import create_logger

logger = create_logger(__name__)


class MidiToPythonVariablePipeline(pipeline.Pipeline):

    def __init__(self, name="Read midis to python variable"):
        """Class used for loading midi files into ppython variable

        :param: (string) name: Pipeline name

        """
        super(MidiToPythonVariablePipeline, self).__init__(
            input_type=Iterable,
            output_type=str,
            name=name)

        self.stat1 = statistics.Counter('how_many_mid_files')
        self.stats = [self.stat1]

    def transform(self, folder, seed_int=666):
        # To make sure that training is always on the same data i used seed
        """Read all the midi files from folder to python variable

        :param: (string) folder: path to folder where midi files are located
        :return: python list of all midi files
        :rtype: list        
        """
        list_all_midi = glob.glob(folder)
        seed(seed_int)
        shuffle(list_all_midi)
        for _ in range(100):
            self.stat1.increment()
        logger.debug("Read {} midi files from location {}".format(
            str(self.stat1), folder))
        return list_all_midi

    def get_stats(self):
        """Returns stats about pipeline
        """
        return self.stats
