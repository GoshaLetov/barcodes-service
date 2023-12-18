APP_PORT := 5050
APP_HOST := '0.0.0.0'

.PHONY: start
start:
	python -m uvicorn src.app:main --host=$(APP_HOST) --port=$(APP_PORT)

.PHONY: download
download:
	dvc pull models/ocr/resnet50.onnx.dvc
	dvc pull models/segmentation/efficientnet0.onnx.dvc

.PHONY: build
build:
	docker build --tag $(DOCKER_IMAGE):$(DOCKER_TAG) .

# PORTS: 4000 -> $(APP_PORT)
.PHONY: run
run:
	docker run \
		--detach \
		--publish 4000:$(APP_PORT) \
		--name k.khvoshchev.hw2.service.5000 \
		--restart always \
		$(DOCKER_IMAGE):$(DOCKER_TAG)

.PHONY: login
login:
	docker login -u $(CI_REGISTRY_USER) -p $(CI_REGISTRY_PASSWORD) $(CI_REGISTRY)

.PHONY: pull
pull:
	docker pull $(DOCKER_IMAGE):latest || true

.PHONY: deploy
deploy:
	ansible-playbook -i deploy/inventory.ini deploy/deploy.yml \
		-e docker_image=$(DOCKER_IMAGE) \
		-e docker_tag=$(DOCKER_TAG) \
		-e docker_registry=$(CI_REGISTRY) \
		-e docker_registry_user=$(CI_REGISTRY_USER) \
		-e docker_registry_password=$(CI_REGISTRY_PASSWORD)

.PHONY: destroy
destroy:
	ansible-playbook -i deploy/inventory.ini deploy/destroy.yml

.PHONY: lint
lint:
	flake8 src/

.PHONY: tests_unit
tests_unit:
	PYTHONPATH=. pytest tests/unit

.PHONY: tests_integration
tests_integration:
	PYTHONPATH=. pytest tests/integration

.PHONY: tests
tests:
	make tests_integration
	make tests_unit
