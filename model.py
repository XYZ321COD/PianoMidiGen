"""Includes all deep learning architecutes I used
"""

import tensorflow as tf
from tensorflow.keras import backend as K
import yaml
file = open(r'settings_model.yaml')
settings = yaml.load(file, Loader=yaml.FullLoader)


def create_model_smaller_emb(seq_len, unique_notes, dropout=settings['dropout'], output_emb=settings['emb_unit'], rnn_unit=settings['rnn_unit'], dense_unit=settings['dense_unit']):
    """
    Creates Deep learning model with 3 layer of GRU
    """
    inputs = tf.keras.layers.Input(shape=(seq_len))
    embedding = tf.keras.layers.Embedding(
        input_dim=unique_notes+1, output_dim=output_emb, input_length=seq_len)(inputs)
    forward_pass = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(rnn_unit, return_sequences=True))(embedding)
    forward_pass = tf.keras.layers.Dropout(dropout)(forward_pass)
    forward_pass = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(rnn_unit, return_sequences=True))(forward_pass)
    forward_pass = tf.keras.layers.Dropout(dropout)(forward_pass)
    forward_pass = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(rnn_unit))(forward_pass)
    forward_pass = tf.keras.layers.Dropout(dropout)(forward_pass)
    forward_pass = tf.keras.layers.Dense(dense_unit)(forward_pass)
    forward_pass = tf.keras.layers.LeakyReLU()(forward_pass)
    outputs = tf.keras.layers.Dense(
        unique_notes+1, activation="softmax")(forward_pass)

    model = tf.keras.Model(inputs=inputs, outputs=outputs,
                           name='Piano_AI_GRU_3_Times_smaller')
    return model
