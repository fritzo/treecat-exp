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
	python main.py --datasets=housing --models=fancysvd,treecat --parallel --log-errors

cleanup-credit: FORCE
	python main.py --datasets=credit --models=fancysvd,treecat --parallel -j 4 --log-errors

cleanup-news: FORCE
	python main.py --datasets=news --models=fancysvd,treecat --parallel -j 2 --log-errors

train-credit: FORCE
	python train.py --dataset=credit -lr=0.01 -ar=0.01 -b=2000 --cuda -n=100 -c=8
	python train.py --dataset=credit -lr=0.01 -ar=0.01 -b=2000 --cuda -n=100 -c=12
	python train.py --dataset=credit -lr=0.01 -ar=0.01 -b=2000 --cuda -n=100 -c=16
	python train.py --dataset=credit -lr=0.01 -ar=0.01 -b=2000 --cuda -n=100 -c=24
	python train.py --dataset=credit -lr=0.01 -ar=0.01 -b=2000 --cuda -n=100 -c=32
	python train.py --dataset=credit -lr=0.01 -ar=0.01 -b=2000 --cuda -n=100 -c=48
	python train.py --dataset=credit -lr=0.01 -ar=0.01 -b=2000 --cuda -n=100 -c=64

train-treecat: FORCE
	python train.py --default-config --dataset=housing
	python train.py --default-config --dataset=credit
	python train.py --default-config --dataset=news
	python train.py --default-config --dataset=census
	python train.py --default-config --dataset=lending

experiments: FORCE
	python main.py

clean: FORCE
	find . -name '*.pyc' -delete -o -name '*.log' -delete
	rm -rf data/*.pkl results/*

cleanresults: FORCE
	rm -rf results/*

FORCE:
