# TreeCat experiments

This requires the contrib-treecat branch of Pyro
https://github.com/pyro-ppl/pyro/pull/1370

See paper at
https://github.com/fritzo/treecat-paper

## Organization

-   [treecat_exp/preprocess.py](blob/master/treecat_exp/preprocess.py)
    contains scripts to load and preprocess a few datasets.
    To add a new dataset, please add a `load_my_dataset()` function in this file.

-   [train.py](blob/master/train.py) is the main training script for TreeCat models.
    To train a dataset run e.g.
    ```sh
    python train.py --dataset census --batch-size 8192 --num-epochs 2 --cuda
    ```
    Training is required before evaluating any model.

-   [train.ipynb](blob/master/train.ipynb) and
    [debug_feature_init.ipynb](blob/master/debug_feature_init.ipynb)
    are notebooks to assess training convergence and diagnose issues with initialization.

-   [evaluate.py](blob/master/evaluate.py) and
    [evaluate.ipynb](blob/master/evaluate.ipynb) evaluate imputation accuracy based on
    posterior predictive probability of true data conditioned on observed data.

-   [eval_predictor.py](blob/master/eval_predictor.py) and
    [eval_predictor.ipynb](blob/master/eval_predictor.ipynb)
    evaluate the accuracy of a downstream logistic regression model that is trained on
    imputed data with various amounts of completely random missingness.

-   [Makefile](blob/master/Makefile) contains convenience commands.
    Please try to `make test` before pushing changes :smile:

### Potential datasets

- [UCI datastes](https://archive.ics.uci.edu/ml/datasets.php)
- Kaggle datasets:
  [medium](https://www.kaggle.com/datasets?sortBy=votes&group=public&page=1&pageSize=20&size=medium&filetype=all&license=all),
  [large](https://www.kaggle.com/datasets?sortBy=votes&group=public&page=1&pageSize=20&size=large&filetype=all&license=all)

- [Kaggle Lending club](https://www.kaggle.com/wendykan/lending-club-loan-data)
- [Kaggle DonorsChoose](https://www.kaggle.com/donorschoose/io#Donors.csv)
- [UCI US Census 1990](https://archive.ics.uci.edu/ml/datasets/US+Census+Data+%281990%29)
- [UCI Letter Recognition](https://archive.ics.uci.edu/ml/datasets/Letter+Recognition)
- [UCI Spambase](https://archive.ics.uci.edu/ml/datasets/Spambase)
- [UCI Default of credit card](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
- [UCI Online news](https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity)
