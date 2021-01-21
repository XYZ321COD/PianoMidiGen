from tensorflow.keras.optimizers import Nadam
import utils.pipelines as df
import utils.project_utils.noteToken as nt
from trainer import TrainModel
import os
from model import create_model_smaller_emb
import tensorflow as tf
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Nadam
import pickle
import datetime
import yaml
from utils.project_utils.logger_ import create_logger
logger = create_logger(__name__)

file = open(r'settings_train.yaml')
settings = yaml.load(file, Loader=yaml.FullLoader)

list_all_midi = df.pipeline1.MidiToPythonVariablePipeline().transform(
    settings['path_to_dataset'])
logger.info("Training on CPU")

sampled_midi = list_all_midi[0:settings['number_of_songs_to_train_on']]
note_tokenizer = nt.NoteTokenizer()
seq_len = settings['seq_len']

logger.info("sequence lenght {} and {} songs to train on".format(
    seq_len, settings['number_of_songs_to_train_on']))

dict_time_notes = df.pipeline2.MidiFilesToPianoRoll().transform(
    sampled_midi, settings['frame_per_seconds'])
full_notes = df.pipeline3.PianoRollsToIntoNoteAndTimeSteps(
).transform(dict_time_notes, seq_len=seq_len, fp=settings['frame_per_seconds'])
for note in full_notes:
    note_tokenizer.add_all_notes(list(note.values()))

note_tokenizer.add_empty_note('empty')
model = create_model_smaller_emb(seq_len, note_tokenizer.unique_notes)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

epochs = settings['epochs']
batch_song = settings['batch_song']
batch_piano_rolls = settings['batch_sequences']
number_of_songs_to_train_on = settings['number_of_songs_to_train_on']
frame_per_second = settings['frame_per_seconds']

optimizer = tf.keras.optimizers.get(settings['optimizer'])
optimizer.lr = settings['lr']
loss_fn = tf.keras.losses.get(settings['loss_function'])

checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 model=model)
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
train_class = TrainModel(epochs, note_tokenizer, full_notes, frame_per_second,
                         batch_piano_rolls, batch_song, optimizer, checkpoint, loss_fn,
                         checkpoint_prefix, number_of_songs_to_train_on, model, seq_len, train_summary_writer)

train_class.train()
model.save(settings['path_to_save_models_to'] + 'model' + model.name +
           str(settings['epochs']) + '_' + str(settings['number_of_songs_to_train_on']) + '.h5')
pickle.dump(note_tokenizer, open(
    settings['path_to_save_models_to'] + "tokenizer" + model.name + str(settings['epochs']) + '_' + str(settings['number_of_songs_to_train_on']) + ".p", "wb"))
