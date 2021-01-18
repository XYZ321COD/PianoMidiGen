from magenta.pipelines import pipeline
from magenta.pipelines import statistics
import IPython
from collections.abc import Iterable
import numpy as np
from ..project_utils.logger_ import create_logger

logger = create_logger(__name__, log_to_console=False)


class NotesAndTimesStepIntoDeepLearningInput(pipeline.Pipeline):

    # Can optimize code a little bit
    def __init__(self, name="Transform format of notes and times step into deep learning input"):
        """Transform format of notes and times step into deep learning input"

        :param: (string) name: Pipeline name.
        """
        super(NotesAndTimesStepIntoDeepLearningInput, self).__init__(
            input_type=Iterable,
            output_type=Iterable,
            name=name)

        self.stat1 = statistics.Counter('how_many_deep_learning_inputs')
        self.stats = [self.stat1]

    def transform(self, dict_keys_time, seq_len=50):
        """ Generate input and the target of our deep learning for one music

        :param: (dict) dict_keys_time : Dictionary of timestep and notes
        :param: (int) seq_len : The length of the sequence
        :return: list of all midi files but in format that can be accepted by neural network - input, target
        :rtype: tuple(list, list)
        """
        collected_list_input, collected_list_target = [], []
        for song in dict_keys_time:
            start_time, end_time = list(song.keys())[0], list(song.keys())[-1]
            list_training, list_target = [], []
            for index_enum, time in enumerate(range(start_time, end_time)):
                list_append_training, list_append_target = [], []
                start_iterate = 0
                # Songs starts with given note, so we must create seq of empty notes which ends with this note
                if index_enum < seq_len:
                    start_iterate = seq_len - index_enum - 1
                    for _ in range(start_iterate):
                        list_append_training.append('empty')
                # Appends every notes before current note form range (0, seq_len)
                for i in range(start_iterate, seq_len):
                    index_enum = time - (seq_len - i - 1)
                    if index_enum in song:
                        list_append_training.append(
                            ','.join(str(x) for x in song[index_enum]))
                    else:
                        list_append_training.append('empty')

                if time+1 in song:
                    list_append_target.append(
                        ','.join(str(x) for x in song[time+1]))
                else:
                    list_append_target.append('empty')
                list_training.append(list_append_training)
                list_target.append(list_append_target)
            collected_list_input += list_training
            collected_list_target += list_target
        logger.info("Example input {} and output {}, format".format(
            collected_list_input[120], collected_list_target[120]))
        return collected_list_input, collected_list_target

    def get_stats(self):
        """Returns stats about pipeline
        """
        return self.stats
