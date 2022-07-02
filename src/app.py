
from fastapi import BackgroundTasks
from fastapi import FastAPI
from src.config import trial_number, models_dir, package_dir
from src.pipeline import ClcikhouseToTfformat, PipeLine
from src.train import TrainModel
import gc
import json
import logging
import os
import time
import uvicorn
from src.model import DNN

tags_metadata = [
    {
        "name": "Model",
        "description": "Operations for the end to end",
    },
    {
        "name": "Default",
        "description": "Basic function for API"
    }
]

app = FastAPI(
    title="Tensorflow Project Demo",
    description="This is machine learning project API",
    version="0.0.1",
    openapi_tags=tags_metadata)

@app.get("/model/metrics", tags=["Model"])
def get_model_metrics():
    """Get the model metrics"""
    models_metrics = []
    for directory in os.listdir(models_dir):
        try:
            with open(
                    os.path.join(
                        models_dir,
                        directory,
                        "metrics.json"), "r") as f:
                model_metrics = json.load(f)
        except Exception as e:
            logging.error("Error in geting model metrics: {}".format(e))
        models_metrics.append(
            {"model_name": directory, "metrics": model_metrics})
    return models_metrics

@app.put("/model", tags=["Model"])
async def retrain_model(background_tasks: BackgroundTasks):
    """Retrain the model"""
    def task_retrain_model():

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
            best_model = train_keras_model.get_best_model(
                trial_number= trial_number)
            # best_model = train_keras_model.simple_train()
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

    background_tasks.add_task(task_retrain_model)

    return {"train": "True"}



@app.get('/ping', tags=["Default"])
def ping():
    return "pong"


if __name__ == "__main__":
    uvicorn.run(app=app)
