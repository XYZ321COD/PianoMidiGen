import os
import webbrowser
import utils.pipelines as df
import utils.project_utils.noteToken as nt
from tqdm import tqdm
from model import create_model
import tensorflow as tf
from numpy.random import choice
import pickle
import numpy as np
import yaml
from utils.project_utils.logger_ import create_logger
import webbrowser
import os

logger = create_logger(__name__)
logger.info("Started evaluation")

file = open(r'settings_eval.yaml')
settings_eval = yaml.load(file, Loader=yaml.FullLoader)

model = tf.keras.models.load_model(
    "models/" + "model" + settings_eval['model_prefix'] + settings_eval['path_to_model'])
note_tokenizer = pickle.load(
    open("models/" + "tokenizer" + settings_eval['model_prefix'] + settings_eval['path_to_tokenizer'], "rb"))

_, seq_len = model.input_shape
_, output_shape = model.output_shape

logger.info("Input of the model has shape {} and Output is {}".format(
    seq_len, output_shape))


def generate_notes(model, unique_notes, max_generated=10000, seq_len=10):
    generate = [note_tokenizer.notes_to_index['empty']
                for i in range(seq_len-1)]
    generate += [note_tokenizer.notes_to_index['empty']]
    for i in tqdm(range(max_generated)):
        test_input = np.array([generate])[:, i:i+seq_len]
        predicted_note = model.predict(test_input)
        random_note_pred = choice(
            unique_notes+1, 1, replace=False, p=predicted_note[0])
        generate.append(random_note_pred[0])
    return generate


generate = generate_notes(
    model, note_tokenizer.unique_notes, settings_eval['song_length'], seq_len)
df.pipeline6.ModelOutPutIntoMidiFile().transform(
    note_tokenizer, generate, "out_put.mid", start_index=seq_len-1, fs=settings_eval['frame_per_seconds'],
    max_generated=settings_eval['song_length'])

webbrowser.open('file://' + os.path.realpath('./midi-visualizer/index.html'))
