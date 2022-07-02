import sys
sys.path.append('..')

from src.model import DNN
from src.pipeline import PipeLine, ClcikhouseToTfformat
import numpy as np
import tensorflow as tf


class TestKerasModel:
    def test_create_model(self):
        sql2tfrecord = ClcikhouseToTfformat()
        data = sql2tfrecord.get_data()
        formated_data, mapping_dict = sql2tfrecord.format_data(data)

        print(mapping_dict)
        model = DNN(mapping_dict)
        model = model.build()



        # model = keras_model.create_model(
        #     learning_rate=0.05,
        #     dense_1=1,
        #     dense_2=1)

        # assert round(
        #     float(
        #         tf.keras.backend.eval(model.optimizer.lr)),
        #     3) == 0.05
        assert len(model.layers) == 37