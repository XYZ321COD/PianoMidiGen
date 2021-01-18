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
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.losses import sparse_categorical_crossentropy
from trainer import TrainModel
import datetime
import re


logger = create_logger(__name__)
logger.info("Started evaluation")

file = open(r'settings_finetune.yaml')
settings_finetune = yaml.load(file, Loader=yaml.FullLoader)


model = tf.keras.models.load_model(
    "models/" + "model" + settings_finetune['model_prefix'] + settings_finetune['path_to_model'])
note_tokenizer = pickle.load(
    open("models/" + "tokenizer" + settings_finetune['model_prefix'] + settings_finetune['path_to_tokenizer'], "rb"))

_, seq_len = model.input_shape
_, output_shape = model.output_shape

logger.info("Input of the model has shape {} and Output is {}".format(
    seq_len, output_shape))


list_all_midi = df.pipeline1.MidiToPythonVariablePipeline().transform(
    settings_finetune['path_to_dataset'])
logger.info("Training on CPU")

sampled_2_midi = list_all_midi[0:settings_finetune['number_of_songs_to_train_on']]
logger.info("sequence lenght {} and {} songs to train on".format(
    seq_len, settings_finetune['number_of_songs_to_train_on']))

dict_time_notes = df.pipeline2.MidiFilesToPianoRoll().transform(
    sampled_2_midi, settings_finetune['frame_per_seconds'])
full_notes = df.pipeline3.PianoRollsToIntoNoteAndTimeSteps(
).transform(dict_time_notes, seq_len=seq_len, fp=settings_finetune['frame_per_seconds'])

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

m = re.search(r'[^_]*', settings_finetune['path_to_model'])
PREV_AMOUT_OF_EPOCH = int(m.group(0))
EPOCHS = settings_finetune['epochs']
BATCH_SONG = settings_finetune['batch_song']
BATCH_NNET_SIZE = settings_finetune['batch_sequences']
TOTAL_SONGS = settings_finetune['number_of_songs_to_train_on']
FRAME_PER_SECOND = settings_finetune['frame_per_seconds']
optimizer = Nadam()

checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 model=model)
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
loss_fn = sparse_categorical_crossentropy

train_class = TrainModel(EPOCHS, note_tokenizer, full_notes, FRAME_PER_SECOND,
                         BATCH_NNET_SIZE, BATCH_SONG, optimizer, checkpoint, loss_fn,
                         checkpoint_prefix, TOTAL_SONGS, model, seq_len, train_summary_writer)

train_class.train()

model.save(settings_finetune['path_to_save_models_to'] + 'model' + model.name +
           str(PREV_AMOUT_OF_EPOCH + settings_finetune['epochs']) + '_' + str(settings_finetune['number_of_songs_to_train_on']) + '.h5')
