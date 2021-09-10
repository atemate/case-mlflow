ENTRYPOINT = ../entrypoint.sh

install:
	pip install -r requirements.txt

train:
	$(ENTRYPOINT) python src/train.py

predict:
	$(ENTRYPOINT) python src/predict.py

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
	    -d @input_example.json

clean:
	rm -r ./mlruns*