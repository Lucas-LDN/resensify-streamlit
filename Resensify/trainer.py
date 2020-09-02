import os
import multiprocessing
import time
import math
import warnings
from tempfile import mkdtemp
from google.cloud import storage
import joblib
import mlflow
import pandas as pd
from Resensify.data import get_data, clean_df
from Resensify.params import NUM_LABELS, BERT_URL, BUCKET_NAME
from Resensify.encoders import BertEncoder, BertClassifier
from Resensify.utils import simple_time_tracker
from Resensify.gcp import storage_upload

from memoized_property import memoized_property
from mlflow.tracking import MlflowClient
from psutil import virtual_memory
from termcolor import colored
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics import confusion_matrix


# Mlflow wagon server
#MLFLOW_URI = "https://mlflow.lewagon.co/"
BATCH_SIZE = 32

class Trainer(object):
    # Mlflow parameters identifying the experiment, you can add all the parameters you wish
    ESTIMATOR = "BERT"
    EXPERIMENT_NAME = "Resensify"

    def __init__(self, Data, **kwargs):
        """
        FYI:
        __init__ is called every time you instatiate Trainer
        Consider kwargs as a dict containig all possible parameters given to your constructor
        Example:
            TT = Trainer(nrows=1000, estimator="Linear")
               ==> kwargs = {"nrows": 1000,
                            "estimator": "Linear"}
        :param X:
        :param y:
        :param kwargs:
        """
        self.text_model = None
        self.kwargs = kwargs
        self.local = kwargs.get("local", True)  # if True training is done locally
        self.mlflow = kwargs.get("mlflow", False)  # if True log info to nlflow
        self.experiment_name = kwargs.get("experiment_name", self.EXPERIMENT_NAME)  # cf doc above
        self.epochs = kwargs.get("epochs", 1)  # if True training is done locally
        self.batch_size = kwargs.get("batch_size", 16)  # if True training is done locally
        self.batched_dataset = Data
        self.nrows = kwargs.get("nrows", 100)
        self.vocab_length = kwargs.get("vocab_length", 10000)
        del Data
        #self.log_kwargs_params()
        #self.log_machine_specs()


    @simple_time_tracker
    def train(self):
        tic = time.time()
        TOTAL_BATCHES = self.nrows // self.batch_size + 1
        TEST_BATCHES = TOTAL_BATCHES // 10
        self.batched_dataset.shuffle(TOTAL_BATCHES)
        test_data = self.batched_dataset.take(TEST_BATCHES)
        train_data = self.batched_dataset.skip(TEST_BATCHES)
        EMB_DIM = 256
        CNN_FILTERS = 128
        DNN_UNITS = 256
        DROPOUT_RATE = 0.2

        self.text_model = BertClassifier(vocabulary_size=self.vocab_length,
                                embedding_dimensions=EMB_DIM,
                                cnn_filters=CNN_FILTERS,
                                dnn_units=DNN_UNITS,
                                model_output_classes=NUM_LABELS,
                                dropout_rate=DROPOUT_RATE)

        if NUM_LABELS == 2:
            self.text_model.compile(loss="binary_crossentropy",
                               optimizer="adam",
                               metrics=["accuracy"])
        else:
            self.text_model.compile(loss="sparse_categorical_crossentropy",
                               optimizer="adam",
                               metrics=["sparse_categorical_accuracy"])

        self.text_model.fit(train_data, validation_data = test_data, batch_size=self.batch_size, epochs=self.epochs)

        # Set up epochs and steps
        # mlflow logs
        #self.mlflow_log_metric("train_time", int(time.time() - tic))

    #def evaluate(self):
        #y_pred = self.predict(self.X_train)
        #tn, fp, fn, tp = confusion_matrix(y_pred, self.y_train.numpy()).ravel()
        #acc_train = (tp+tn)/(tp+tn+fn+fp)
        #self.mlflow_log_metric("accuracy_train", acc_train)


    #def predict(self, X = None):
        #result = self.bert_classifier(X, training=False)
        #result_type = tf.argmax(result, 1).numpy()
        #return result_type

    def save_model(self, upload=False, auto_remove=False):
        """Save the model into a .joblib and upload it on Google Storage /models folder
        HINTS : use sklearn.joblib (or jbolib) libraries and google-cloud-storage"""
        #joblib.dump(self.bert_classifier, 'model.joblib')
        #self.bert_classifier.save(path)

        #joblib.dump(self.text_model, 'model.joblib')
        path = "my_model/{}".format(NUM_LABELS)
        self.text_model.save(path)
        print(colored("model saved locally", "green"))
        #tf.saved_model.save(self.bert_classifier, export_dir=export_dir)
        #print(colored("model saved locally", "green"))
        if upload:
          storage_upload()




    ### MLFlow methods
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        if self.mlflow:
            self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        if self.mlflow:
            self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    def log_estimator_params(self):
        reg = self.get_estimator()
        self.mlflow_log_param('estimator_name', reg.__class__.__name__)
        params = reg.get_params()
        for k, v in params.items():
            self.mlflow_log_param(k, v)

    def log_kwargs_params(self):
        if self.mlflow:
            for k, v in self.kwargs.items():
                self.mlflow_log_param(k, v)

    def log_machine_specs(self):
        cpus = multiprocessing.cpu_count()
        mem = virtual_memory()
        ram = int(mem.total / 1000000000)
        self.mlflow_log_param("ram", ram)
        self.mlflow_log_param("cpus", cpus)


if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    # Get and clean data
    experiment = "Resensify"
    params = dict(nrows=100,
                  upload=True, # upload model.job lib to strage if set to True
                  local=True,
                  epochs = 1,
                  batch_size = BATCH_SIZE,
                  mlflow=False,  # set to True to log params to mlflow
                  experiment_name=experiment)
    print("############   Loading Data   ############")
    df = get_data(**params)
    df = clean_df(df)
    b = BertEncoder()
    processed_dataset= b.transform(df)
    vocab_length = b.get_vocab_length()
    params['vocab_length'] = vocab_length
    batched_dataset = processed_dataset.padded_batch(BATCH_SIZE, padded_shapes=((None, ), ()))
    del df
    # Train and save model, locally and
    t = Trainer(Data = batched_dataset, **params)
    print(colored("############  Training model   ############", "red"))
    t.train()
    print(colored("############  Evaluating model ############", "blue"))
    # t.evaluate()
    print(colored("############   Saving model    ############", "green"))
    #t.save_model(upload=False)
    del batched_dataset, processed_dataset
