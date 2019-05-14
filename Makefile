.PHONY: all test FORCE

all: test

install: FORCE
	pip install -r requirements.txt

lint: FORCE
	flake8

test: FORCE lint
	python main.py -n 1

#clean: FORCE
#	find . -name '*.pyc' -delete -o -name '*.log' -delete
#	rm -f data/*.pkl model/*

FORCE:
