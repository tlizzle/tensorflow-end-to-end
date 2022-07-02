import logging
import gc
from src.config import package_dir, models_dir
from src.pipeline import PipeLine, ClcikhouseToTfformat
from src.train import TrainModel
import time
import os
from src.model import DNN

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def main():
    try:
        logging.info("Query data from database")
        sql2tfrecord = ClcikhouseToTfformat()
        data = sql2tfrecord.get_data()
        formated_data, mapping_dict = sql2tfrecord.format_data(data)
        del data
        gc.collect()

        sql2tfrecord.write2tfrecord(
            data=formated_data,
            filename= os.path.join(
                package_dir, "data", "data.tfrecord"))

    except Exception as e:
        logging.error("Error in process data to tfrecord: {}".format(e))
        return 0
    logging.info("Writed data to tfrecord")

    try:
        logging.info("Start initializing training process")
        pipeline = PipeLine(
            tf_filename= os.path.join(
                package_dir, "data", "data.tfrecord"))

        model = DNN(mapping_dict)
        train_keras_model = TrainModel(pipeline, model)

    except Exception as e:
        logging.error(
            "Error in initializing training process: {}".format(e))
        return 0

    try:
        logging.info("Start Searching Best Model")
        # best_model = train_keras_model.get_best_model(
        #     trial_number= 5)
        best_model = train_keras_model.simple_train()
    except Exception as e:
        logging.error("Error in searching best model: {}".format(e))
        return 0

    try:
        logging.info("Start saving model")
        result = train_keras_model.save_model(
            model=best_model,
            filename=os.path.join(
                models_dir, str(int(time.time()))))
    except Exception as e:
        logging.error("Error in saving model: {}".format(e))

    logging.critical("Retrain Finish. Training result: {}".format(result))


if __name__ == "__main__":
    main()