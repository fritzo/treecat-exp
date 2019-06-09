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
	python -O main.py --datasets=housing --models=fancysvd,treecat16,treecat32,treecat64 --parallel -j=7 --log-errors

cleanup-credit: FORCE
	python -O main.py --datasets=credit --models=fancysvd,treecat16,treecat32,treecat64 --log-errors

cleanup-news: FORCE
	python -O main.py --datasets=news --models=fancysvd,treecat16,treecat32,treecat64 --log-errors

train-credit: FORCE
	python -O train.py --cuda  --dataset=credit -b=2000 -n=100 -c=8
	python -O train.py --cuda  --dataset=credit -b=2000 -n=100 -c=12
	python -O train.py --cuda  --dataset=credit -b=2000 -n=100 -c=16
	python -O train.py --cuda  --dataset=credit -b=2000 -n=100 -c=24
	python -O train.py --cuda  --dataset=credit -b=2000 -n=100 -c=32
	python -O train.py --cuda  --dataset=credit -b=2000 -n=100 -c=48
	python -O train.py --cuda  --dataset=credit -b=2000 -n=100 -c=64

train-news: FORCE
	python -O train.py --cuda  --dataset=credit -b=2048 -n=100 -c=8
	python -O train.py --cuda  --dataset=credit -b=2048 -n=100 -c=12
	python -O train.py --cuda  --dataset=credit -b=2048 -n=100 -c=16
	python -O train.py --cuda  --dataset=credit -b=2048 -n=100 -c=24
	python -O train.py --cuda  --dataset=credit -b=2048 -n=100 -c=32
	python -O train.py --cuda  --dataset=credit -b=2048 -n=100 -c=48
	python -O train.py --cuda  --dataset=credit -b=2048 -n=100 -c=64

train-treecat: FORCE
	python -O train.py --default-config --dataset=housing
	python -O train.py --default-config --dataset=credit
	python -O train.py --default-config --dataset=news
	python -O train.py --default-config --dataset=census
	python -O train.py --default-config --dataset=lending

experiments: FORCE
	$(MAKE) cleanup-housing
	$(MAKE) cleanup-credit
	$(MAKE) cleanup-news
	$(MAKE) train-credit
	$(MAKE) train-news
	$(MAKE) train-treecat
	@echo DONE

clean: FORCE
	find . -name '*.pyc' -delete -o -name '*.log' -delete
	rm -rf data/*.pkl results/*

cleanresults: FORCE
	rm -rf results/*

FORCE:
