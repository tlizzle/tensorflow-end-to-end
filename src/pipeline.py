import tensorflow as tf
from clickhouse_driver import Client
from src.encoder import Encoder
import logging
from src.config import categorical_features, label_name, host
import pandas as pd

class ClcikhouseToTfformat(object):
    """
    load the data from clickhouse(db) and convert to tfRecord format 
    for further training
    """

    def int64_feature(self, value):
        return tf.train.Feature(
            int64_list=tf.train.Int64List(value=[value])
        )

    def bytes_feature(self, value):
        return tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[value.encode()])
        )
        
    def float_feature(self, value):
        return tf.train.Feature(
            float_list=tf.train.FloatList(value=[value])
        )

    def get_data(self):
        try:
            client = Client(host= host, 
                            port= 9000, 
                            user= 'user1', 
                            password= '123456',
                            settings={'use_numpy': True})
            data = client.query_dataframe('SELECT * FROM testing.winprice_estimation')
            data.drop('date', inplace= True, axis= 1)
            client.disconnect()

        except Exception as err:
            logging.error(f'Fail to get data from db: {err}')
            return 0
        return data


    def format_data(self, data):
        """Format the query data from the Clickhouse"""
        # data[label_name] = list(
        #     map(lambda x: predict_categories.index(x), data["variety"]))
        # data[label_name] = tf.keras.utils.to_categorical(data[label_name])

        encoder = Encoder(data).fit(categorical_features)
        output = encoder.transform()
        mapping_dict = encoder.id_encodding
        output.columns = categorical_features

        data = data.loc[:, ~data.columns.isin(categorical_features)]
        data = pd.concat([data, output], axis= 1)

        return data, mapping_dict


    def write2tfrecord(self, data, filename):
        """Write the formated data into tfrecord file."""
        with tf.io.TFRecordWriter(filename) as writer:
            for i in range(len(data[list(data.keys())[0]])):
                feature = {
                    "price": self.float_feature(data["price"][i]),
                    "bedrooms": self.float_feature(data["bedrooms"][i]),
                    "bathrooms": self.float_feature(data["bathrooms"][i]),
                    "sqft_living": self.int64_feature(data["sqft_living"][i]),
                    "sqft_lot": self.int64_feature(data["sqft_lot"][i]),
                    "floors": self.float_feature(data["floors"][i]),
                    "waterfront": self.int64_feature(data["waterfront"][i]),
                    "view": self.int64_feature(data["view"][i]),
                    "condition": self.int64_feature(data["condition"][i]),
                    "sqft_above": self.int64_feature(data["sqft_above"][i]),
                    "sqft_basement": self.int64_feature(data["sqft_basement"][i]),    
                    "yr_built": self.int64_feature(data["yr_built"][i]),
                    "yr_renovated": self.int64_feature(data["yr_renovated"][i]),    
                    "street": self.int64_feature(data["street"][i]),    
                    "city": self.int64_feature(data["city"][i]),    
                    "statezip": self.int64_feature(data["statezip"][i]),
                    "country": self.int64_feature(data["country"][i]), 
                    "weekday": self.int64_feature(data["weekday"][i])                                                                                                                                                                                                                                                                             
                }
                example = tf.train.Example(
                    features=tf.train.Features(feature=feature))
                example = example.SerializeToString()
                writer.write(example)


class PipeLine(object):
    """Load and transform data from tfrecords.
    
    Attributes:
        tfrecords_filenames: tfrecord file names in list
    """
        
    def __init__(self, tf_filename):
        self.features = {
            "price": tf.io.FixedLenFeature([], tf.float32),
            "bedrooms": tf.io.FixedLenFeature([], tf.float32),
            "bathrooms": tf.io.FixedLenFeature([], tf.float32),
            "sqft_living": tf.io.FixedLenFeature([], tf.int64),
            "sqft_lot":  tf.io.FixedLenFeature([], tf.int64),
            "floors": tf.io.FixedLenFeature([], tf.float32),
            "waterfront":  tf.io.FixedLenFeature([], tf.int64),
            "view":  tf.io.FixedLenFeature([], tf.int64),
            "condition":  tf.io.FixedLenFeature([], tf.int64),
            "sqft_above":  tf.io.FixedLenFeature([], tf.int64),
            "sqft_basement":  tf.io.FixedLenFeature([], tf.int64),    
            "yr_built":  tf.io.FixedLenFeature([], tf.int64),
            "yr_renovated":  tf.io.FixedLenFeature([], tf.int64),    
            "street":  tf.io.FixedLenFeature([], tf.int64),    
            "city": tf.io.FixedLenFeature([], tf.int64),    
            "statezip": tf.io.FixedLenFeature([], tf.int64),
            "country": tf.io.FixedLenFeature([], tf.int64), 
            "weekday": tf.io.FixedLenFeature([], tf.int64) 
        }


        full_dataset = tf.data.TFRecordDataset(tf_filename)
        data_size = 0
        for _ in full_dataset:
            data_size += 1

        train_size = int(0.7 * data_size)
        test_size = int(0.1 * data_size)
        val_size = int(0.2 * data_size)


        full_dataset = full_dataset.shuffle(buffer_size=1)

        full_dataset = full_dataset.map(self.parse_data)
        self.train_dataset = full_dataset.take(train_size)
        test_dataset = full_dataset.skip(train_size)
        self.val_dataset = test_dataset.skip(val_size)
        self.test_dataset = test_dataset.take(test_size)

    def parse_data(self, serialized_object):
        """Format tfrecord data.
        
        Args:
            serialized: The record in the tfrecord.
        
        Returns:
            Formated record.
        """
        parsed_example = tf.io.parse_example(serialized_object, self.features)

        inputs = {}
        for key in parsed_example.keys():
            inputs[key] = parsed_example[key]
        return (inputs, {'price': parsed_example['price']})

    def get_train_data(self, batch_size):
        return self.train_dataset.batch(batch_size)

    def get_val_data(self, batch_size):
        return self.val_dataset.batch(batch_size)

    def get_test_data(self, batch_size):
        return self.test_dataset.batch(batch_size)

if __name__ == "__main__":
    pass
    # sql2tfrecord = ClcikhouseToTfformat()
    # data = sql2tfrecord.get_data()
    # formated_data, mapping_dict = sql2tfrecord.format_data(data)


    # sql2tfrecord.write2tfrecord(
    #     data= formated_data,
    #     filename= "data.tfrecord")

    # pipeline = PipeLine("data.tfrecord")
    # pipeline.train_dataset

