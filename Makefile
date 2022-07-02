IMAGE_NAME=tensorflow-end-to-end:latest

build:
	docker build -t ${IMAGE_NAME} . 

activate-tensorboard:
	tensorboard --logdir ./logs/tunning --host 0.0.0.0 

run-app-dev:
	uvicorn src.app:app --reload 

run-app:
	uvicorn src.app:app --host 0.0.0.0 --port 8000

init-sql:
	python -m scripts.insert_data_to_db

deploy-default-model:
	cp -r tensorflow-end-to-end/* /opt/tensorflow-end-to-end

init-training: init-sql deploy-default-model run-app


serving-health-check:
	curl http://localhost:8501/v1/models/tensorflow-end-to-end

predict:
	curl -X POST "http://localhost:8501/v1/models/tensorflow-end-to-end:predict" -d '{"inputs":{"bedrooms":[[0]],"bathrooms":[[0]],"sqft_living":[[0]],"sqft_lot":[[0]],"floors":[[0]],"waterfront":[[0]],"view":[[0]],"condition":[[0]],"sqft_above":[[0]],"sqft_basement":[[0]],"yr_built":[[0]],"yr_renovated":[[0]],"street":[[0]],"city":[[0]],"statezip":[[0]],"country":[[0]],"weekday":[[0]]}}' -H  "accept: application/json"

