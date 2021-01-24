import os
import webbrowser
import utils.pipelines as df
import utils.project_utils.noteToken as nt
from tqdm import tqdm
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


generate = df.pipeline7.GenMusic().transform(
    model, note_tokenizer, note_tokenizer.unique_notes, settings_eval['song_length'], seq_len)
df.pipeline6.ModelOutPutIntoMidiFile().transform(
    note_tokenizer, generate, "out_put.mid", start_index=seq_len-1, fs=settings_eval['frame_per_seconds'],
    max_generated=settings_eval['song_length'])

webbrowser.open('file://' + os.path.realpath('./midi-visualizer/index.html'))
