import os
import tensorflow as tf
from src.config import categorical_features, continuous_features
from src.pipeline import PipeLine, ClcikhouseToTfformat
from kerastuner import HyperModel
import keras_tuner as kt

METRICS = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'), 
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(curve='ROC', name='roc'),
      tf.keras.metrics.AUC(curve='PR', name='pr')
      ]

package_dir = os.path.dirname(os.path.abspath("__file__"))

class DNN(HyperModel):
    def __init__(self, mapping_dict):
        self.mapping_dict = mapping_dict
        self.emm_dim = {
            "street": 100,
            "city": 45,
            "statezip": 50,
            "country": 1,
            "weekday": 7
        }

    def build(self, hp= None, training= None):
        inputs = {}
        concatenated_feature = []
        for feature_name in continuous_features:
            if feature_name != 'price':
                inputs[feature_name] = tf.keras.layers.Input(shape= (1),\
                                                name= feature_name)
                concatenated_feature.append(inputs[feature_name])
        if hp:
            for cat in categorical_features:
                n_cat = len(self.mapping_dict[cat])
                inputs[cat] = tf.keras.layers.Input(shape=(1,),name= cat)

                embed = tf.keras.layers.Embedding(input_dim= n_cat,\
                                                output_dim= hp.Int(f"{cat}_emb", min_value=5, max_value=100, step=10),\
                                                trainable= True, \
                                                embeddings_initializer= tf.keras.initializers.GlorotNormal(), \
                                                name= cat + '_emb')
                embedding_lookup_weight = tf.keras.layers.Reshape(
                    (-1,))(embed(inputs[cat]))
                concatenated_feature.append(embedding_lookup_weight)

            pre_preds= tf.keras.layers.concatenate(concatenated_feature)

            for i in range(hp.Int("num_layers", 1, 3)):
                pre_preds = tf.keras.layers.Dense(
                                units= hp.Int(f"units_{i}", min_value=8, max_value=126, step=32),
                                activation= hp.Choice("activation", ["relu", "tanh"]),
                                name= f"units_{i}",
                            )(pre_preds)
                if hp.Boolean("dropout"):
                    pre_preds = tf.keras.layers.Dropout(
                                        rate= 0.2, name= f'dropout_{i}')(
                                            pre_preds, training=training)

            pred = tf.keras.layers.Dense(1, activation="linear")(pre_preds)

            model = tf.keras.Model(inputs, {'price': pred})

            model.compile(loss=tf.keras.losses.MeanSquaredError(),\
                            metrics=['mean_squared_error'],
                            optimizer= 'adam')   
        else:
            for cat in categorical_features:
                n_cat = len(self.mapping_dict[cat])
                inputs[cat] = tf.keras.layers.Input(shape=(1,),name= cat)

                embed = tf.keras.layers.Embedding(input_dim= n_cat,\
                                                output_dim= self.emm_dim[cat],\
                                                trainable=True, \
                                                embeddings_initializer=tf.keras.initializers.GlorotNormal(), \
                                                name= cat + '_emb')
                embedding_lookup_weight = tf.keras.layers.Reshape(
                    (self.emm_dim[cat],))(embed(inputs[cat]))
                concatenated_feature.append(embedding_lookup_weight)

            pre_preds= tf.keras.layers.concatenate(concatenated_feature)

            pre_preds = tf.keras.layers.BatchNormalization()(pre_preds)
            pre_preds = tf.keras.activations.tanh(pre_preds)
            
            pre_preds = tf.keras.layers.BatchNormalization()(pre_preds)
            pre_preds = tf.keras.layers.Dense(units=32,
                                            activation= tf.nn.relu,
                                            kernel_initializer= tf.keras.initializers.GlorotNormal(),
                                        )(pre_preds)
            pre_preds = tf.keras.layers.Dropout(.2)(pre_preds)

            pre_preds = tf.keras.layers.BatchNormalization()(pre_preds)
            pre_preds = tf.keras.layers.Dense(units=16,
                                            activation= tf.nn.relu,
                                            kernel_initializer= tf.keras.initializers.GlorotNormal(),
                                        )(pre_preds)
            pre_preds = tf.keras.layers.Dropout(.2)(pre_preds)

            pred = tf.keras.layers.Dense(1, activation="linear")(pre_preds)

            model = tf.keras.Model(inputs, {'price': pred})

            model.compile(loss=tf.keras.losses.MeanSquaredError(),\
                            metrics=['mean_squared_error'],
                            optimizer= 'adam')   
        return model


if __name__ == "__main__":
    pass
    # sql2tfrecord = ClcikhouseToTfformat()
    # data = sql2tfrecord.get_data()
    # formated_data, mapping_dict = sql2tfrecord.format_data(data)

    # sql2tfrecord.write2tfrecord(
    #     data=formated_data,
    #     filename= os.path.join("data", "data.tfrecord")
    # )
    # pipeline = PipeLine(os.path.join("data", "data.tfrecord"))
    # train_dataset = pipeline.get_train_data(1000)
    # val_dataset = pipeline.get_val_data(1000)
    # model = DNN(mapping_dict)
    # model = model.build()

    # model.fit(
    #     train_dataset,
    #     validation_data=val_dataset,
    #     verbose= 1,
    #     epochs= 10,
    #     batch_size= 32,
    #     workers=10,
    # )
            
    # tuner = kt.RandomSearch(
    #     model,
    #     objective='val_loss',
    #     max_trials=5,
    #     overwrite=True,
    #     executions_per_trial=2
    # )

    # tuner.search(train_dataset, 
    #                 batch_size= 32, 
    #                 epochs=10, 
    #                 validation_data= val_dataset, 
    #                 workers=8)
    
    # best_model = tuner.get_best_models(1)
