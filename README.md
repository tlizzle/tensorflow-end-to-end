# Tensorflow end-to-end Demo

The purpose of this project is to demonstrate and deliver the minimal of end-to-end Machine Learning project in real life. 

In this project the dataset is about house price estimation and this project contains three service
1. Database
    - When the request of model training / retraining is received, the `training` service will get the data from the databases which is clickhouse in this case
2. Training
    - Get the data from `Database` and start the training process
3. Serving
    - After the completion of training process, the model is saved for serving purpose meaning any request can be sent to this service and get prediction result


With the purpose of demonstrating how data-related porjects are developed in real-world scenarios, efforts have been contributed on following:
1. Files for deployment: `Dockerfile` and `docker-compose.yml`
2. Core application written in python in `src`
3. Necessary scipts in `scripts/`
4. Configuration setting in `src/config.py`

## Tensorflow
Some powerfull tools often used with tensorflow
1. tfrecord
2. tensorboard
3. tensorflow serving


## Services

### Database
I use clickhouse  in this project so as the trainining request has been received `Training` will get the training data from clickhouse. The following command can show what the data look likes
```
docker-compose exec clickhouse  clickhouse-client --user user1 --password 123456 --query "select * from testing.winprice_estimation LIMIT 10"
```

### Training
The training platform uses fastApi with the port on 8000 which could be connected to (http://localhost:8000/docs).

The retraining process start from getting the data from `Databases` and write the data on local as tfrecord format.
The benefits of tfrecord format are memory usage reduction during training and data storage reduction.

### Serving
The native tensorflow serving service has been selected as it is very powerful tool (https://www.tensorflow.org/tfx/serving/api_rest)


## How to run this project
1. Prerequisite 
    - docker
    - docker-compose

2. Image preparation
    - `make build`

3. Services activation
    - `docker-compose up -d`

4. Model retrain (optional)
    - `curl -X PUT "http://localhost:8000/model" -H  "accept: application/json"`

5. Check training result
     - `docker-compose exec training make activate-tensorboard`
     - browse the link (http://localhost:6006/)

6. Prediction
```
curl -X POST "http://localhost:8501/v1/models/tensorflow-end-to-end:predict" -d '{"inputs":{"bedrooms":[[0]],"bathrooms":[[0]],"sqft_living":[[0]],"sqft_lot":[[0]],"floors":[[0]],"waterfront":[[0]],"view":[[0]],"condition":[[0]],"sqft_above":[[0]],"sqft_basement":[[0]],"yr_built":[[0]],"yr_renovated":[[0]],"street":[[0]],"city":[[0]],"statezip":[[0]],"country":[[0]],"weekday":[[0]]}}' -H  "accept: application/json"
{
    "outputs": [
        [
            426926.781
        ]
    ]
}
```
The output indicates the house winprice estiation

# Run (dev mode)
- If virtual env already exists, activate: pipenv shell
    - If not, create virtual env: pipenv shell
    - Install all required packages:
        - install packages exactly as specified in Pipfile.lock: pipenv sync
        - install using the Pipfile, including the dev packages: pipenv install --dev

## Misc
Model structure and hyperparameter definition adjustment could be made in `src/model.py` as
im using HyperModel of kerastuner to serarch the best set of hyperparameter in search space.

# Reference
- [kerasTuner](https://keras.io/keras_tuner/)
- [Data science project folder structure](https://dzone.com/articles/data-science-project-folder-structure)
