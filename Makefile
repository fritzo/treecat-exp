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

#clean: FORCE
#	find . -name '*.pyc' -delete -o -name '*.log' -delete
#	rm -f data/*.pkl model/*

FORCE:
