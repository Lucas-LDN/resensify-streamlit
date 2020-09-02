import os

from google.cloud import storage
from termcolor import colored

from Resensify.params import BUCKET_NAME, MODEL_NAME, NUM_LABELS


def storage_upload(bucket=BUCKET_NAME, rm=False):
    client = storage.Client().bucket(bucket)
    storage_location = 'models/{}/versions/{}/{}'.format(
        MODEL_NAME,
        NUM_LABELS,
        'model.joblib')
    blob = client.blob(storage_location)
    blob.upload_from_filename('model.joblib')
    print(colored("=> model.joblib uploaded to bucket {} inside {}".format(BUCKET_NAME, storage_location),
                  "green"))
    if rm:
        os.remove('model.joblib')
