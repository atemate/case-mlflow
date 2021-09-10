ENTRYPOINT = ../entrypoint.sh
INPUT_EXAMPLE = input_example.json

install:
	pip install -r requirements.txt

train:
	$(ENTRYPOINT) python src/train.py

predict:
	$(ENTRYPOINT) python src/predict.py --json-file=$(INPUT_EXAMPLE)

promote:
	$(ENTRYPOINT) python src/promote-model.py

ui:
	$(ENTRYPOINT) mlflow server \
		--backend-store-uri '$${MLFLOW_TRACKING_URI}' \
		--default-artifact-root ./mlruns

serve:
	$(ENTRYPOINT) mlflow models serve \
		--no-conda \
		-p 8080 \
		-m 'models:/$${MY_MODEL_NAME}/$${MY_MODEL_STAGE}'

payload:
	curl -v http://127.0.0.1:8080/invocations \
		-H 'Content-Type: application/json' \
	    -d @$(INPUT_EXAMPLE)

clean:
	rm -r ./mlruns*