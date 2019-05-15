.PHONY: all test FORCE

all: test

install: FORCE
	pip install -r requirements.txt

dirs: FORCE
	mkdir -p data results/train results/test

lint: FORCE
	flake8

test: FORCE lint dirs
	python train.py -n 1

#clean: FORCE
#	find . -name '*.pyc' -delete -o -name '*.log' -delete
#	rm -f data/*.pkl model/*

FORCE:
