# Predicting Insurance Policy Renewals
Top 1 solution to the [TBrain - 客戶續約金額預測](https://tbrain.trendmicro.com.tw/Competitions/Details/3) machine learning competition.

This is the exact same code used to generate the best submission.

[Solution Documentation (in traditional Chinese)](docs/solution_documentation.md)

## EDA (Exploratory Data Analysis) Notebook

Located inside the [notebooks](notebooks/) folder.

Quick links:
 * [HTML](<notebooks/EDA - Public.html>)
 * [ipynb](<notebooks/EDA - Public.ipynb>)

## Model Training and Submission Generation

Potential Compatibility Issue: I use [fish shell](https://fishshell.com/). If you're using Bash, you might need to change `set -x SEED num; ` to `SEED=num `.

### 1st Layer

#### Train Full LGBM Regression Model
```
set -x SEED 9989; python simple_lgb.py
```

#### Train LGBM Regression Models with Limited Features

(The last parameter represents the number of best features to be used in training)

```
set -x SEED 11511; python truncate_lgb.py 50
set -x SEED 13511; python truncate_lgb.py 40
set -x SEED 12511; python truncate_lgb.py 60
```

#### Train Full LGBM Classification Model
```
set -x SEED 1989; python lgb_binary_target.py
```

#### Train LGBM Classification Models with Limited Features

(The last parameter represents the number of best features to be used in training)

```
set -x SEED 22511; python lgb_binary_truncate.py 50
set -x SEED 23513; python lgb_binary_truncate.py 40
set -x SEED 52511; python lgb_binary_truncate.py 60
```

### 2nd Layer

#### Train LGB Ensembles

```
set -x SEED 51; python ensemble_lgb.py 50
set -x SEED 15; python ensemble_lgb.py 50
```

#### Train DNN Ensembles

```
set -x SEED 515; python ensemble_dnn.py
set -x SEED 151; python ensemble_dnn.py
```

### 3rd Layer - DNN Ensemble

```
set -x SEED 1515; python ensemble_dnn_2nd.py
set -x SEED 5151; python ensemble_dnn_2nd.py
```

### Create Final Submission By Averaging

```
python avg_files.py
```

The **final submission file** (prediction to the test dataset) can be found in the project root folder with file name `sub_ens.csv`.

**Note: The final model has a CV loss of 1657.81, which is slightly higher than the one of the best submission (1656.24). This is because the numbers of models in the 1st and 2nd layers are reduced to save training time.**
