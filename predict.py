import os
from math import sqrt

import joblib
import pandas as pd
import numpy as np
from tensorflow import keras
from termcolor import colored

from Resensify.trainer import BATCH_SIZE
from Resensify.params import BUCKET_NAME, NUM_LABELS
from Resensify.encoders import BertEncoder
from google.cloud import storage

def get_test_data():
    """method to get the training data (or a portion of it) from google cloud bucket
    To predict we can either obtain predictions from train data or from test data"""
    # Add Client() here
    path = "data/test.csv"
    df = pd.read_csv(path)
    return df


def download_model_gcp():
    client = storage.Client().bucket(BUCKET_NAME)

    storage_location1 = 'models/saved_model.pb'
    storage_location2 = 'models/variables/variables.index'
    storage_location3 = 'models/variables/variables.data-00000-of-00001'

    blob = client.blob(storage_location1)
    blob.download_to_filename('gcp_model/saved_model.pb')

    blob = client.blob(storage_location2)
    blob.download_to_filename('gcp_model/variables/variables.index')

    blob = client.blob(storage_location3)
    blob.download_to_filename('gcp_model/variables/variables.data-00000-of-00001')

    print(f"=> pipeline downloaded from storage")
    model = keras.models.load_model("gcp_model")
    return model

def download_model_local(num_labels=NUM_LABELS):
    path = "my_model/{}".format(num_labels)
    model = keras.models.load_model(path)
    return model

#def generate_predictions():
#    print(colored("############  Downloading test data   ############", "red"))
#    df_test = get_test_data()
#    print(df_test[["text", "review_type"]])
#    print(colored("############  Downloading model   ############", "red"))
#    model = download_model_local()
#    # Check if model savec was the ouptut of RandomSearch or Gridsearch
#    b = BertEncoder()
#    X =  b.transform(df_test)
#    X_batch = X.padded_batch(BATCH_SIZE, padded_shapes=((None, ), ()))
#    y_pred = model.predict(X_batch)
#    print(y_pred)
#    if NUM_LABELS == 2:
#      review_type = np.zeros_like(y_pred)
#      review_type[y_pred>0.5] = 1
#    else:
#      review_type = np.argmax(y_pred, axis=1)
#    print(review_type)
#

def generate_predictions_from_list(fake_reviews, num_labels = NUM_LABELS):
  fake_scores = [0]*len(fake_reviews)
  df = pd.DataFrame(
    {'text': fake_reviews,
     'review_type': fake_scores
    })
  model = download_model_local(num_labels)
  b = BertEncoder()
  X =  b.transform(df)
  X_batch = X.padded_batch(BATCH_SIZE, padded_shapes=((None, ), ()))
  y_pred = model.predict(X_batch)
  if num_labels == 2:
     review_type = np.zeros_like(y_pred)
     review_type[y_pred>0.5] = 1
  elif num_labels == 5:
     review_type = np.argmax(y_pred, axis=1)+1
  return review_type.astype(int).ravel()


if __name__ == '__main__':
    fake_reviews = ['Service was horrible, I hate this place', 'Amazing experience. I will be back!']
    print(fake_reviews)
    review_type = generate_predictions_from_list(fake_reviews,5)
    print(review_type)
