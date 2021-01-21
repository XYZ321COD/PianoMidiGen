"""
Module defining how deep learning models are trained
"""
import utils.pipelines as df
from tqdm import tqdm
import tensorflow as tf
import numpy as np

from utils.project_utils.logger_ import create_logger


logger = create_logger(__name__, log_to_file=False)


class TrainModel:
    """Class responsible of training deep learning models

    :param: (int) epochs: how many epochs
    :param: (noteToken) note_tokenizer: instance of class holding information about training process
    :param: (sampled_midi) midi_files : list of midi_files
    :param: (int) frame_per_second: number of frame per seconds
    :param: (int) batch_sequences : size of batch of sequences
    :param: (int) batch_song : size of batch of songs
    :param: (keras.optimizer) optimizer - type of optimizer used
    :param: (tensorflow.checkpoint) checkpoint - used for saving data about training
    :param: (keras.loss_function) loss_fn - type of loss function used
    :param: (tensorflow.checkpoint_prefix) checkpoint_prefix - prefix to checkpoint
    :param: (int) total_songs - number of songs model will be trained on
    :param: (tensorflow.model) model - deep learning model
    :param: (int) seq_len  - size of sequences
    :param: (tensorflow.train_summary_writer) train_summary_writer - used for saving data about traning process

    """

    def __init__(self, epochs, note_tokenizer, sampled_midi, frame_per_second,
                 batch_sequences, batch_song, optimizer, checkpoint, loss_fn,
                 checkpoint_prefix, total_songs, model, seq_len, train_summary_writer):
        self.epochs = epochs
        self.note_tokenizer = note_tokenizer
        self.sampled_midi = sampled_midi
        self.frame_per_second = frame_per_second
        self.batch_sequences = batch_sequences
        self.batch_song = batch_song
        self.optimizer = optimizer
        self.checkpoint = checkpoint
        self.loss_fn = loss_fn
        self.checkpoint_prefix = checkpoint_prefix
        self.total_songs = total_songs
        self.model = model
        self.seq_len = seq_len
        self.train_summary_writer = train_summary_writer

    def train(self):
        for epoch in tqdm(range(self.epochs), desc='Training epochs'):
            for count, i in enumerate(tqdm(range(0, self.total_songs, self.batch_song), desc='Iterating over batch musics midi')):
                loss_total = 0
                inputs_nnet_large, outputs_nnet_large = df.pipeline4.NotesAndTimesStepIntoDeepLearningInput(
                ).transform(self.sampled_midi[i:i+self.batch_song], seq_len=self.seq_len)
                inputs_nnet_large = np.array(
                    self.note_tokenizer.transform(inputs_nnet_large), dtype=np.int32)
                outputs_nnet_large = np.array(
                    self.note_tokenizer.transform(outputs_nnet_large), dtype=np.int32)

                index_shuffled = np.arange(
                    start=0, stop=len(inputs_nnet_large))
                np.random.shuffle(index_shuffled)

                for nnet_steps in tqdm(range(0, len(index_shuffled), self.batch_sequences)):
                    current_index = index_shuffled[nnet_steps:nnet_steps +
                                                   self.batch_sequences]
                    inputs_nnet, outputs_nnet = inputs_nnet_large[
                        current_index], outputs_nnet_large[current_index]

                    if len(inputs_nnet) // self.batch_sequences != 1:
                        break
                    loss = self.train_step(
                        inputs_nnet, outputs_nnet) / self.batch_sequences
                    loss_total += tf.math.reduce_sum(loss)
                logger.info("epochs {} | Batch {} | total loss : {}".format(
                    epoch + 1, count, loss_total))
            with self.train_summary_writer.as_default():
                tf.summary.scalar('Loss', loss_total /
                                  self.batch_song, epoch)  # add summary
                self.train_summary_writer.flush()
        self.checkpoint.save(file_prefix=self.checkpoint_prefix)

    @ tf.function
    def train_step(self, inputs, targets):
        """One step of training process

        :param: (np.ndarray) inputs: inputs of model
        :param: (np.ndarray) targets: targets of model
        :return: value of the loss function
        :rtype: float

        """
        with tf.GradientTape() as tape:
            prediction = self.model(inputs)
            loss = self.loss_fn(targets, prediction)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))
        return loss
