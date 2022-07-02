from src.pipeline import PipeLine, ClcikhouseToTfformat
import json
import os
import tensorflow as tf
import keras_tuner as kt



class TrainModel(object):
    """Train the tensorflow keras model.
    Attributes:
        pipeline: pipeline object.
    """
    
    def __init__(self, pipeline, model):
        self.pipeline = pipeline
        self.model = model

        self.train_dataset = self.pipeline.get_train_data(32)
        self.val_dataset = self.pipeline.get_val_data(32)

    def simple_train(self):
        """Fit the model with the data from pipeline
        Args:
            hp: hyperparameters in dictionary format.
        """

        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = self.model.build()

        logdir = os.path.join("logs", 'tunning')
        tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, update_freq='batch', histogram_freq=1)

        model.fit(
            self.train_dataset,
            validation_data= self.val_dataset,
            verbose=1,
            epochs=10,
            callbacks= [tensorboard_callback]
            )
            
        return model

    def get_best_model(self, trial_number):
        """Fit the model with data and find the best hyperparameter space
        in the search space using kerastune.
        Args:
            trial_number: number of model that keras tune will train.
        
        Returns:
            The best keras model within all the search space.
        """
        model = self.model
        tuner = kt.RandomSearch(
            model,
            objective='val_loss',
            max_trials= trial_number,
            overwrite=True,
            executions_per_trial=2
        )
        logdir = os.path.join("logs", 'tunning')
        tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, update_freq='batch', histogram_freq=1)

        tuner.search(self.train_dataset, 
                        epochs=10, 
                        validation_data= self.val_dataset, 
                        workers=8,
                        callbacks= [tensorboard_callback]
                        )
        
        models = tuner.get_best_models(num_models=2)
        best_model = models[0]
        return best_model

    def save_model(self, model, filename):
        """Save the model for the serving usage and
        also write the performance metrics.
        """
        test_dataset = self.pipeline.get_test_data(batch_size=10)
        metrics = model.evaluate(test_dataset)

        model.save(filename, save_format= 'tf')
        res = {
            "loss": float(metrics[0]),
            "mse": float(metrics[1])}

        with open(os.path.join(filename, "metrics.json"), "w") as f:
            json.dump(res, f)
        return res

if __name__ == "__main__":
    pass
    # sql2tfrecord = ClcikhouseToTfformat()
    # data = sql2tfrecord.get_data()
    # formated_data, mapping_dict = sql2tfrecord.format_data(data)

    # sql2tfrecord.write2tfrecord(
    #     data=formated_data,
    #     filename= os.path.join("data", "data.tfrecord"))

    # model = DNN(mapping_dict)

    # pipeline = PipeLine(os.path.join("data", "data.tfrecord"))
    # tm = TrainModel(pipeline, model)
    # # tm.get_best_model(5)
    # tm.simple_train()


        
