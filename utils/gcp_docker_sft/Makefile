PROJECT_NAME := orpo-sft

all: build

.PHONY: build

build:
	docker build --platform=linux/amd64 -t $(PROJECT_NAME):latest .
	docker tag $(PROJECT_NAME) us-central1-docker.pkg.dev/sudm-ai/testing/$(PROJECT_NAME):latest
	docker push us-central1-docker.pkg.dev/sudm-ai/testing/$(PROJECT_NAME):latest

.PHONY: job
job:
	kubectl create -f ./job.yaml

.PHONY: desc
des:
	kubectl describe job $(PROJECT_NAME)
