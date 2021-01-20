"""Includes all deep learning architecutes I used
"""

import tensorflow as tf
from tensorflow.keras import backend as K


def create_model_smaller_emb(seq_len, unique_notes, dropout=0.0, output_emb=256, rnn_unit=50, dense_unit=256):
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


def create_model(seq_len, unique_notes, dropout=0.0, output_emb=2000, rnn_unit=128, dense_unit=2000):
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
                           name='Piano_AI_GRU_3_Times')
    return model


def create_model_deep2times(seq_len, unique_notes, dropout=0.3, output_emb=100, rnn_unit=128, dense_unit=2000):
    """
    Creates Deep learning model with 6 layer of GRU
    """
    inputs = tf.keras.layers.Input(shape=(seq_len,))
    embedding = tf.keras.layers.Embedding(
        input_dim=unique_notes+1, output_dim=output_emb, input_length=seq_len)(inputs)
    forward_pass = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(rnn_unit, return_sequences=True))(embedding)
    forward_pass = tf.keras.layers.Dropout(dropout)(forward_pass)
    forward_pass = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(rnn_unit, return_sequences=True))(forward_pass)
    forward_pass = tf.keras.layers.Dropout(dropout)(forward_pass)
    forward_pass = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(rnn_unit, return_sequences=True))(forward_pass)
    forward_pass = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(rnn_unit, return_sequences=True))(forward_pass)
    forward_pass = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(rnn_unit))(forward_pass)
    forward_pass = tf.keras.layers.Dropout(dropout)(forward_pass)
    forward_pass = tf.keras.layers.Dense(dense_unit)(forward_pass)
    forward_pass = tf.keras.layers.LeakyReLU()(forward_pass)
    outputs = tf.keras.layers.Dense(
        unique_notes+1, activation="softmax")(forward_pass)

    model = tf.keras.Model(inputs=inputs, outputs=outputs,
                           name='Piano_AI_GRU_6_Times')
    return model


def create_model_2TimesWider(seq_len, unique_notes, dropout=0.3, output_emb=100, rnn_unit=256, dense_unit=2000):
    """
    Creates Deep learning model with 3 layer of GRU but more rnn_units
    """
    inputs = tf.keras.layers.Input(shape=(seq_len,))
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
                           name='Piano_AI_GRU_3_Times_2Units')
    return model


def create_model_attention(seq_len, unique_notes, dropout=0.3, output_emb=200, rnn_unit=512, dense_unit=1000):
    inputs = tf.keras.layers.Input(shape=(seq_len,))
    embedding = tf.keras.layers.Embedding(
        input_dim=unique_notes+1, output_dim=output_emb, input_length=seq_len)(inputs)
    forward_pass = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(rnn_unit, return_sequences=True))(embedding)
    forward_pass, att_vector = SeqSelfAttention(
        return_attention=True,
        attention_activation='sigmoid',
        attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
        attention_width=50,
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        bias_regularizer=tf.keras.regularizers.l1(1e-4),
        attention_regularizer_weight=1e-4,
    )(forward_pass)
    forward_pass = tf.keras.layers.Dropout(dropout)(forward_pass)
    forward_pass = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(rnn_unit, return_sequences=True))(forward_pass)
    forward_pass, att_vector2 = SeqSelfAttention(
        return_attention=True,
        attention_activation='sigmoid',
        attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
        attention_width=50,
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        bias_regularizer=tf.keras.regularizers.l1(1e-4),
        attention_regularizer_weight=1e-4,
    )(forward_pass)
    forward_pass = tf.keras.layers.Dropout(dropout)(forward_pass)
    forward_pass = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(rnn_unit))(forward_pass)
    forward_pass = tf.keras.layers.Dropout(dropout)(forward_pass)
    forward_pass = tf.keras.layers.Dense(dense_unit)(forward_pass)
    forward_pass = tf.keras.layers.LeakyReLU()(forward_pass)
    outputs = tf.keras.layers.Dense(
        unique_notes+1, activation="softmax")(forward_pass)
    model = tf.keras.Model(
        inputs=inputs, outputs=outputs, name='attention_model')
    return model
