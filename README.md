# TreeCat experiments

This requires the contrib-treecat branch of Pyro
https://github.com/pyro-ppl/pyro/pull/1370

See paper at
https://github.com/fritzo/treecat-paper

## Datasets

[treecat_exp/preprocess.py](treecat_exp/preprocess.py)
contains scripts to load and preprocess a few datasets.
To add a new dataset, please add a `load_my_dataset()` function in this file.

Existing datasets include:
- [Boston housing](http://lib.stat.cmu.edu/datasets/boston)
- [UCI Online news](https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity)
- [UCI US Census 1990](https://archive.ics.uci.edu/ml/datasets/US+Census+Data+%281990%29)
- [Kaggle Lending club](https://www.kaggle.com/wendykan/lending-club-loan-data)

Potential new datasets include:
- [UCI datastes](https://archive.ics.uci.edu/ml/datasets.php)
- Kaggle datasets:
  [medium](https://www.kaggle.com/datasets?sortBy=votes&group=public&page=1&pageSize=20&size=medium&filetype=all&license=all),
  [large](https://www.kaggle.com/datasets?sortBy=votes&group=public&page=1&pageSize=20&size=large&filetype=all&license=all)
- [Kaggle DonorsChoose](https://www.kaggle.com/donorschoose/io#Donors.csv)
- [UCI Letter Recognition](https://archive.ics.uci.edu/ml/datasets/Letter+Recognition)
- [UCI Spambase](https://archive.ics.uci.edu/ml/datasets/Spambase)
- [UCI Default of credit card](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)

## Training

[train.py](train.py) is the main training script for TreeCat models.
To train a dataset run e.g.
```sh
python train.py --dataset census --batch-size 8192 --num-epochs 2 --cuda
```
Training is required before evaluating any model.

[train.ipynb](train.ipynb) and
[debug_feature_init.ipynb](debug_feature_init.ipynb)
are notebooks to assess training convergence and diagnose issues with initialization.

## Evaluation / Experiments

[evaluate.py](evaluate.py) and
[evaluate.ipynb](evaluate.ipynb) evaluate imputation accuracy based on
posterior predictive probability of true data conditioned on observed data.

[eval_predictor.py](eval_predictor.py) and
[eval_predictor.ipynb](eval_predictor.ipynb)
evaluate the accuracy of a downstream logistic regression model that is trained on
imputed data with various amounts of completely random missingness.

## Testing

[Makefile](Makefile) contains convenience commands.
Please try to `make test` before pushing changes :smile:
