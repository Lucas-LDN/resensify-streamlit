import os

import pandas as pd
import numpy as np

import tensorflow as tf

import tensorflow_hub as hub

from tensorflow.keras import layers
from official.nlp import bert
import official.nlp.bert.tokenization

from sklearn.base import BaseEstimator, TransformerMixin
from Resensify.utils import tokenize_review
from Resensify.data import get_data, clean_df
from Resensify.params import GS_FOLDER_BERT, NUM_LABELS, BERT_URL
from google.cloud import storage


class BertEncoder(BaseEstimator, TransformerMixin):
    # class TimeFeaturesEncoder(CustomEncoder):

    def __init__(self):
        #client = storage.Client().bucket(GS_FOLDER_BERT)
        #blob = client.blob('vocab.txt')
        #vocab = blob.download_as_string()
        bert_layer = hub.KerasLayer(BERT_URL, trainable=False)
        vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
        to_lower = bert_layer.resolved_object.do_lower_case.numpy()
        self.tokenizer = bert.tokenization.FullTokenizer(vocab_file, to_lower)

    def transform(self, df):
        reviews = list(df.text)
        tokenized_reviews = [tokenize_review(review, self.tokenizer) for review in reviews]
        sorted_reviews = sorted(zip(tokenized_reviews, df.review_type.tolist()), key = len)
        processed_dataset = tf.data.Dataset.from_generator(lambda: sorted_reviews,
                              output_types=(tf.int32, tf.int32))
        return processed_dataset

    def get_vocab_length(self):
        return len(self.tokenizer.vocab)

    def fit(self, X, y=None):
        return self

class BertClassifier(tf.keras.Model):

    def __init__(self,
                 vocabulary_size,
                 embedding_dimensions=128,
                 cnn_filters=50,
                 dnn_units=512,
                 model_output_classes=2,
                 dropout_rate=0.1,
                 training=False,
                 name="text_model"):
        super(BertClassifier, self).__init__(name=name)

        self.embedding = layers.Embedding(vocabulary_size,
                                          embedding_dimensions)
        self.cnn_layer1 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=2,
                                        padding="valid",
                                        activation="relu")
        self.cnn_layer2 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=3,
                                        padding="valid",
                                        activation="relu")
        self.cnn_layer3 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=4,
                                        padding="valid",
                                        activation="relu")
        self.pool = layers.GlobalMaxPool1D()

        self.dense_1 = layers.Dense(units=dnn_units, activation="relu")
        self.dropout = layers.Dropout(rate=dropout_rate)
        if model_output_classes == 2:
            self.last_dense = layers.Dense(units=1,
                                           activation="sigmoid")
        else:
            self.last_dense = layers.Dense(units=model_output_classes,
                                           activation="softmax")

    def call(self, inputs, training):
        l = self.embedding(inputs)
        l_1 = self.cnn_layer1(l)
        l_1 = self.pool(l_1)
        l_2 = self.cnn_layer2(l)
        l_2 = self.pool(l_2)
        l_3 = self.cnn_layer3(l)
        l_3 = self.pool(l_3)

        concatenated = tf.concat([l_1, l_2, l_3], axis=-1) # (batch_size, 3 * cnn_filters)
        concatenated = self.dense_1(concatenated)
        concatenated = self.dropout(concatenated, training)
        model_output = self.last_dense(concatenated)

        return model_output


if __name__ == "__main__":
    params = dict(nrows=1000,
                  local=True)  # set to False to get data from GCP (Storage or BigQuery)
    df = get_data(**params)
    df = clean_df(df)
    b = BertEncoder()
    X =  b.transform(df)
    print(X)





