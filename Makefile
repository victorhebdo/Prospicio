# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* Prospicio/*.py

black:
	@black scripts/* Prospicio/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr Prospicio-*.dist-info
	@rm -fr Prospicio.egg-info

install:
	@pip install . -U

all: clean install test black check_code

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
PYPI_USERNAME=<AUTHOR>
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u $(PYPI_USERNAME)

pypi:
	@twine upload dist/* -u $(PYPI_USERNAME)

run_api:
	uvicorn prospicio.api.fast:app --reload

#======================#
#         Docker       #
#======================#

# Local images - using local computer's architecture
# i.e. linux/amd64 for Windows / Linux / Apple with Intel chip
#      linux/arm64 for Apple with Apple Silicon (M1 / M2 chip)

DOCKER_IMAGE_NAME=prospicio
DOCKER_LOCAL_PORT=8080
GCR_MULTI_REGION=eu.gcr.io
PROJECT_ID=prospicio-project

docker_build_local:
	docker build --tag=$(DOCKER_IMAGE_NAME):local .

docker_run_local:
	docker run \
		-e PORT=8000 -p $(DOCKER_LOCAL_PORT):8000 \
		$(DOCKER_IMAGE_NAME):local

docker_run_local_interactively:
	docker run -it \
		-e PORT=8000 -p $(DOCKER_LOCAL_PORT):8000 \
		$(DOCKER_IMAGE_NAME):local \
		bash

# Cloud images - using architecture compatible with cloud, i.e. linux/amd64

docker_build:
	docker build \
		--platform linux/amd64 \
		-t $(GCR_MULTI_REGION)/$(PROJECT_ID)/$(DOCKER_IMAGE_NAME):prod .

# Alternative if previous doesn´t work. Needs additional setup.
# Probably don´t need this. Used to build arm on linux amd64
docker_build_alternative:
	docker buildx build --load \
		--platform linux/amd64 \
		-t $(GCR_MULTI_REGION)/$(PROJECT_ID)/$(DOCKER_IMAGE_NAME):prod .

docker_run:
	docker run \
		--platform linux/amd64 \
		-e PORT=8000 -p $(DOCKER_LOCAL_PORT):8000 \
		$(GCR_MULTI_REGION)/$(PROJECT_ID)/$(DOCKER_IMAGE_NAME):prod

docker_run_interactively:
	docker run -it \
		--platform linux/amd64 \
		-e PORT=8000 -p $(DOCKER_LOCAL_PORT):8000 \
		$(GCR_MULTI_REGION)/$(PROJECT_ID)/$(DOCKER_IMAGE_NAME):prod \
		bash

# Push and deploy to cloud

docker_push:
	docker push $(GCR_MULTI_REGION)/$(PROJECT_ID)/$(DOCKER_IMAGE_NAME):prod

docker_deploy:
	gcloud run deploy \
		--project $(PROJECT_ID) \
		--image $(GCR_MULTI_REGION)/$(PROJECT_ID)/$(DOCKER_IMAGE_NAME):prod \
		--platform managed \
		--region europe-west1

# --env-vars-file .env.yaml
#	--env-file .env
