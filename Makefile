MODELS_DIR=commands/misc/tensorflow-models

.PHONY: init
init:
	git submodule update --init
	cd ./commands/misc/tensorflow-models/research && protoc --python_out=. ./object_detection/protos/*.proto
	pip install -r requirements.txt

.PHONY: train
train: init
	# cd nn && python ../commands/misc/tensorflow-models/research/object_detection/legacy/train.py --logtostderr --train_dir=training/ --pipeline_config_path=pipeline.config
	TF_CPP_MIN_LOG_LEVEL=0 \
	PYTHONPATH="$$PYTHONPATH:$(MODELS_DIR)/research:$(MODELS_DIR)/research/slim" \
	python $(MODELS_DIR)/research/object_detection/model_main.py \
		--pipeline_config_path ./nn/pipeline.config \
		--model_dir ./nn/training/model \
		--checkpoint_dir ./nn/training/checkpoint
