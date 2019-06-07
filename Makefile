.PHONY: all test FORCE

all: test

install: FORCE
	pip install -r requirements.txt

lint: FORCE
	flake8

test: FORCE lint
	rm -rf temp.results.test
	RESULTS=temp.results.test python train.py -n 1 -v
	RESULTS=temp.results.test python main.py --smoketest --pdb
	rm -rf temp.results.test
	@echo PASS

# Parallel version of `make test`
ptest: FORCE lint
	rm -rf temp.results.test
	RESULTS=temp.results.test python train.py -n 1 -v
	RESULTS=temp.results.test python main.py --smoketest --parallel
	rm -rf temp.results.test
	@echo PASS

test-vae: FORCE lint
	rm -rf temp.results.test
	RESULTS=temp.results.test python main.py --smoketest --models=vae --pdb
	rm -rf temp.results.test
	@echo PASS

martintest: FORCE lint
	rm -rf temp.results.test
	RESULTS=temp.results.test python main.py --smoketest --models=fancyii,fancysvd,fancyknn
	rm -rf temp.results.test
	@echo PASS

cleanup-housing: FORCE
	python main.py --datasets=housing --parallel --log-errors

cleanup-credit: FORCE
	python main.py --datasets=credit --parallel --log-errors

experiments: FORCE
	python main.py

clean: FORCE
	find . -name '*.pyc' -delete -o -name '*.log' -delete
	rm -rf data/*.pkl results/*

cleanresults: FORCE
	rm -rf results/*

FORCE:
