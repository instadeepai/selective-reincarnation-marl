# Check if GPU is available
NVCC_RESULT := $(shell which nvcc 2> NULL)
NVCC_TEST := $(notdir $(NVCC_RESULT))
ifeq ($(NVCC_TEST),nvcc)
GPUS=--gpus all
else
GPUS=
endif
# For Windows use CURDIR
ifeq ($(PWD),)
PWD := $(CURDIR)
endif
# Set flag for docker run command
BASE_FLAGS=-it --rm  -v ${PWD}:/home/app/rmarl -w /home/app/rmarl
RUN_FLAGS=$(GPUS) $(BASE_FLAGS)
IMAGE=rmarl:latest
RUN_FLAGS_TENSORBOARD=$(GPUS) -p 6006:6006 $(BASE_FLAGS)
DOCKER_RUN=docker run $(RUN_FLAGS) $(IMAGE)
DOCKER_RUN_TPU=docker run $(RUN_FLAGS) --privileged $(IMAGE)
DOCKER_RUN_TENSORBOARD=docker run $(RUN_FLAGS_TENSORBOARD) $(IMAGE)

# Set exp to run when using `make run`
# Default exp
exp=./experiments/default_exp.sh

# make file commands
build:
	docker build --tag $(IMAGE) .

run:
	$(DOCKER_RUN) python $(exp)

run-tpu:
	$(DOCKER_RUN_TPU) python $(exp)

run-tensorboard:
	$(DOCKER_RUN_TENSORBOARD) /bin/bash -c "  tensorboard --bind_all --logdir  /home/app/rmarl/ & $(exp); "

bash:
	$(DOCKER_RUN) bash

bash-tpu:
	$(DOCKER_RUN_TPU) bash

push:
	docker login
	-docker push $(IMAGE)
	docker logout

pull:
	docker pull $(IMAGE)