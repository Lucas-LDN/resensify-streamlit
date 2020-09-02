import pandas as pd
from google.cloud import storage
import re
#import Resensify
from Resensify.params import MAX_LENGTH, NUM_LABELS, BUCKET_NAME, BUCKET_TRAIN_DATA_PATH
from Resensify.utils import simple_time_tracker



@simple_time_tracker
def get_data(nrows=10000, local=True, **kwargs):
    """method to get the training data (or a portion of it) from google cloud bucket"""
    #client = storage.Client()
    if local:
        path = "data/data_sample_AZ_NV.csv"
    else:
        path = "gs://{}/{}".format(BUCKET_NAME, BUCKET_TRAIN_DATA_PATH)
    df = pd.read_csv(path, nrows=nrows)
    return df

@simple_time_tracker
def clean_df(df, test=False):
    df = df[df['text'].notna()]

    if NUM_LABELS == 2:
      dic = {1: 0, 2: 0, 3: 0, 4: 1, 5: 1}
    elif NUM_LABELS == 5:
      dic = {1: 0, 2: 1, 3: 3, 4: 3, 5: 4}

    df['review_type'] = df.stars_x.map(dic)
    df['text'] = df['text'].apply(preprocess_text)
    return df[['text', 'review_type']]



TAG_RE = re.compile(r'<[^>]+>')
def remove_tags(text):
    return TAG_RE.sub('', text)

def preprocess_text(sen):

    sentence = remove_tags(sen)
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence[:MAX_LENGTH]


if __name__ == "__main__":
    params = dict(nrows=10000,
                  upload=False,
                  local=True)  # set to False to get data from GCP
    df = get_data(**params)
    m1 = df.memory_usage().sum()/1000000
    print(m1)
