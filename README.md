# Tabular experiments

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

## Algorithms

- TreeCat (Fritz)
- CrossCat (Fritz)
- VAE variants (JP)
- GAIN (JP)
- MICE for deterministic imputation

Possible other algorithms:
- Multimodal VAE ([Wu & Goodman 2018](https://arxiv.org/abs/1802.05335))
- TreeGauss (Fritz)
- TreeVAE
- other deterministic imputation algorithms

## Evaluation / Experiments

Potential experiments:

-   Imputation.
    ```
    For each dataset:
        For each density in [0.1, 0.2, 0.5, 0.8, 0.9]:
            Randomly remove some density of data (per cell).
            (Note take care to avoid removing entire rows)
            For each algorithm in [treecat, crosscat, VAE, GAN, deterministic]:
                1. impute missing cells; evaluate L1 or L2 loss
                2. evaluate posterior preditive likelihood of missing values
    ```
-   Denoising / outlier detection.
    ```
    For each dataset:
        For each density in [0.001, 0.01, 0.1]:
            Randomly replace some density of data with "bad" values (per cell).
            Note: "bad" might mean "ok in the marginal sense but unexpected
                  when conditioned on rest of row". Or just wrong units e.g.
            For each algorithm in [treecat, crosscat, VAE?, GAN?, ???]:
                1. Predict which cells are outliers
                   Evaluate based on precision/recall curves.
                2. "Clean up" the data and eval L1 or L2 loss wrt truth.
    ```
-   Training a downstream ML algorithm after cleanup.
    Cleanup means (a) imputing missing fields and (b) denoising / outlier removal.
    ```
    For each dataset:
        For each density in [0.1, ..., 0.9]:
            # Dirtying.
            Randomly remove some density of data.
            ?Randomly replace some cells with "bad" values?
            # Cleanup.
            For each algorithm in [treecat, crosscat, vae, gan]:
                Replace all cells with "denoised" version.
                For each column:
                    For each fully-supervised algorithm in
                       [logistic reg., linear reg., xgboost, SVM]:
                       train algo on dirt vs cleaned dataset.
                       Compare trained model on test datset.
    ```
-   Active learning experiment.
    ```
    For each dataset:
        For each density in [0.1, 0.2, 0.5, 0.8, 0.9]:
            Randomly remove some density of data (per cell).
            For each algorithm in [treecat, crosscat, VAE?, GAN?]:
                1. Use algo to suggest (for each row) n cells to observe. (n in {1,2,3})
                2. Add those cells to data (true values).
                3. impute remaining cells; evaluate L1 or L2 loss
                4. evaluate posterior preditive likelihood of missing values
    ```
- Qualitative structure comparison.
    ```
    Pick a representative dataset, e.g. lending club.
    For each algorithm in [treecat, crosscat]:
        Discuss the learned latent structure
    ```

[evaluate.py](evaluate.py) and
[evaluate.ipynb](evaluate.ipynb) evaluate imputation accuracy based on
posterior predictive probability of true data conditioned on observed data.

[eval_predictor.py](eval_predictor.py) and
[eval_predictor.ipynb](eval_predictor.ipynb)
evaluate the accuracy of a downstream logistic regression model that is trained on
imputed data with various amounts of completely random missingness.

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

## Testing

[Makefile](Makefile) contains convenience commands.
Please try to `make test` before pushing changes :smile:
