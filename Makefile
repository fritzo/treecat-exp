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

test-gain: FORCE lint
	rm -rf temp.results.test
	RESULTS=temp.results.test python main.py --smoketest --models=gain --pdb
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

train-housing: FORCE
	python -O train.py --dataset=housing -c 16 -n 100 -b 128 -ar 0.02 -lr 0.01 --suffix=lr_01
	python -O train.py --dataset=housing -c 16 -n 100 -b 128 -ar 0.02 -lr 0.02 --suffix=lr_02
	python -O train.py --dataset=housing -c 16 -n 100 -b 128 -ar 0.02 -lr 0.03 --suffix=lr_03
	python -O train.py --dataset=housing -c 16 -n 100 -b 128 -ar 0.02 -lr 0.04 --suffix=lr_04
	python -O train.py --dataset=housing -c 16 -n 100 -b 128 -ar 0.02 -lr 0.05 --suffix=lr_05
	python -O train.py --dataset=housing -c 16 -n 100 -b 128 -ar 0.02 -lr 0.07 --suffix=lr_07
	python -O train.py --dataset=housing -c 16 -n 100 -b 128 -ar 0.02 -lr 0.10 --suffix=lr_10
	# python -O train.py --dataset=housing -c 16 -n 100 -b 128 -ar 0.02 -lr 0.20 --suffix=lr_20  # NAN
	# python -O train.py --dataset=housing -c 16 -n 100 -b 128 -ar 0.02 -lr 0.50 --suffix=lr_50  # NAN

train-credit: FORCE
	python -O train.py --dataset=credit -b=2000 -n=100 -ar=0.2 -lr=0.3 -c=8
	python -O train.py --dataset=credit -b=2000 -n=100 -ar=0.2 -lr=0.3 -c=12
	python -O train.py --dataset=credit -b=2000 -n=100 -ar=0.2 -lr=0.3 -c=16
	python -O train.py --dataset=credit -b=2000 -n=100 -ar=0.2 -lr=0.3 -c=24
	python -O train.py --dataset=credit -b=2000 -n=100 -ar=0.2 -lr=0.3 -c=32
	python -O train.py --dataset=credit -b=2000 -n=100 -ar=0.2 -lr=0.3 -c=48
	python -O train.py --dataset=credit -b=2000 -n=100 -ar=0.2 -lr=0.3 -c=64

train-news: FORCE
	python -O train.py --dataset=credit -b=2048 -n=100 -c=8
	python -O train.py --dataset=credit -b=2048 -n=100 -c=12
	python -O train.py --dataset=credit -b=2048 -n=100 -c=16
	python -O train.py --dataset=credit -b=2048 -n=100 -c=24
	python -O train.py --dataset=credit -b=2048 -n=100 -c=32
	python -O train.py --dataset=credit -b=2048 -n=100 -c=48
	python -O train.py --dataset=credit -b=2048 -n=100 -c=64

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

clean-vae: FORCE
	find results -name '*vae*' -delete 

clean-gain: FORCE
	find results -name '*gain*' -delete 

clean-treecat: FORCE
	find results -name '*treecat*' -delete 

clean-fancy: FORCE
	find results -name '*fancy*' -delete 

FORCE:
