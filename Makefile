.PHONY: all test FORCE

all: test

install: FORCE
	pip install -r requirements.txt

dirs: FORCE
	mkdir -p data results/train results/test

lint: FORCE
	flake8

test: FORCE lint dirs
	rm -rf temp.results.test
	RESULTS=temp.results.test python train.py -n 1
	RESULTS=temp.results.test python evaluate.py
	@# RESULTS=temp.results.test python eval_predictor.py
	rm -rf temp.results.test
	@echo PASS

results: FORCE
	python train.py --dataset boston_housing -b 64 -n 100 -c 2
	python train.py --dataset boston_housing -b 64 -n 100 -c 4
	python train.py --dataset boston_housing -b 64 -n 100 -c 8
	python train.py --dataset boston_housing -b 64 -n 100 -c 16
	python train.py --dataset news -b 512 -n 20 -c 2
	python train.py --dataset news -b 512 -n 20 -c 4
	python train.py --dataset news -b 512 -n 20 -c 8
	python train.py --dataset news -b 512 -n 20 -c 16
	python train.py --dataset census -b 1024 -n 2 -c 2
	python train.py --dataset census -b 1024 -n 2 -c 4
	python train.py --dataset census -b 1024 -n 2 -c 8
	python train.py --dataset census -b 1024 -n 2 -c 16

clean: FORCE
	find . -name '*.pyc' -delete -o -name '*.log' -delete
	rm -f data/*.pkl results/*

FORCE:
