from magenta.pipelines import pipeline
from magenta.pipelines import statistics
import IPython
from collections.abc import Iterable
import numpy as np
from ..project_utils.logger_ import create_logger
from tqdm import tqdm
import numpy as np

logger = create_logger(__name__, log_to_console=False)


class GenMusic(pipeline.Pipeline):

    # Can optimize code a little bit
    def __init__(self, name="Transform format of notes and times step into deep learning input"):
        """Transform format of notes and times step into deep learning input"

        :param: (string) name: Pipeline name.
        """
        super(GenMusic, self).__init__(
            input_type=Iterable,
            output_type=Iterable,
            name=name)

        self.stat1 = statistics.Counter('how_many_notes')
        self.stats = [self.stat1]

    def transform(self, model, note_tokenizer, unique_notes, max_generated,  seq_len=50):
        generate = [note_tokenizer.notes_to_index['empty']
                    for i in range(seq_len-1)]
        generate += [note_tokenizer.notes_to_index['empty']]
        for i in tqdm(range(max_generated)):
            test_input = np.array([generate])[:, i:i+seq_len]
            predicted_note = model.predict(test_input)
            random_note_pred = np.random.choice(
                unique_notes+1, 1, replace=False, p=predicted_note[0])
            generate.append(random_note_pred[0])
        return generate

    def get_stats(self):
        """Returns stats about pipeline
        """
        return self.stats
