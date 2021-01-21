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
epochs = settings_finetune['epochs']
batch_song = settings_finetune['batch_song']
batch_piano_rolls = settings_finetune['batch_sequences']
number_of_songs_to_train_on = settings_finetune['number_of_songs_to_train_on']
frame_per_second = settings_finetune['frame_per_seconds']

optimizer = tf.keras.optimizers.get(settings_finetune['optimizer'])
optimizer.lr = settings_finetune['lr']
loss_fn = tf.keras.losses.get(settings_finetune['loss_function'])

checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

train_class = TrainModel(epochs, note_tokenizer, full_notes, frame_per_second,
                         batch_piano_rolls, batch_song, optimizer, checkpoint, loss_fn,
                         checkpoint_prefix, number_of_songs_to_train_on, model, seq_len, train_summary_writer)

train_class.train()

model.save(settings_finetune['path_to_save_models_to'] + 'model' + model.name +
           str(PREV_AMOUT_OF_EPOCH + settings_finetune['epochs']) + '_' + str(settings_finetune['number_of_songs_to_train_on']) + '.h5')
