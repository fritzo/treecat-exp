.PHONY: all test FORCE

all: test

install: FORCE
	pip install -r requirements.txt

lint: FORCE
	flake8

test: FORCE lint
	rm -rf temp.results.test
	RESULTS=temp.results.test python train.py -n 1 -v
	RESULTS=temp.results.test python main.py --smoketest
	rm -rf temp.results.test
	@echo PASS

martintest: FORCE lint
	rm -rf temp.results.test
	RESULTS=temp.results.test python main.py --smoketest --models=fancy
	rm -rf temp.results.test
	@echo PASS

train: FORCE
	python train.py --default-config -v --dataset housing
	python train.py --default-config -v --dataset news
	python train.py --default-config -v --dataset census
	python train.py --default-config -v --dataset lending

experiments: FORCE
	python main.py

clean: FORCE
	find . -name '*.pyc' -delete -o -name '*.log' -delete
	rm -rf data/*.pkl results/*

cleanresults: FORCE
	rm -rf results/*

FORCE:
