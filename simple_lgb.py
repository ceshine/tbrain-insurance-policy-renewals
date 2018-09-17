import os
from datetime import date
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

from features import feature_engineering, POLICY_FIXED_CATEGORICALS, EXTRA_CATEGORICALS
from config import KFOLD_SEED, KFOLD_N

MEMORY = joblib.Memory(cachedir="cache/")
OUT_DIR = Path("cache/single/")
OUT_DIR.mkdir(exist_ok=True)
SEED = int(os.environ.get("SEED", 123))


def fit_and_predict(df_train, df_test,  params={}, verbose=True):
    kf = KFold(n_splits=KFOLD_N, random_state=KFOLD_SEED, shuffle=True)
    param_ = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression_l1',
        'feature_fraction': 0.8,
        'bagging_fraction': 0.9,
        'learning_rate': 0.02,
        'bagging_freq': 1,
        'min_data_in_leaf': 100,
        'max_depth': 10,
        'num_threads': 8,
        'seed': SEED,
        'num_leaves': 16,
        'cat_l2': 10,
        'lambda_l1': 0
    }
    param_.update(params)
    val_losses = []
    val_pred_dfs = []
    test_preds = []
    feature_names = df_train.drop(
        "Next_Premium", axis=1).columns.tolist()
    # print(feature_names)
    # print(df_train[feature_names].columns)
    global_importance = defaultdict(int)
    for i, (train_index, val_index) in enumerate(kf.split(df_train)):
        print("-" * 20)
        print(f"Fold {i+1}")
        print("-" * 20)
        train_data = lgb.Dataset(
            df_train.iloc[train_index][feature_names],
            label=df_train.Next_Premium.iloc[train_index],
            categorical_feature=POLICY_FIXED_CATEGORICALS + EXTRA_CATEGORICALS
        )
        valid_data = lgb.Dataset(
            df_train.iloc[val_index][feature_names],
            label=df_train.Next_Premium.iloc[val_index],
            categorical_feature=POLICY_FIXED_CATEGORICALS + EXTRA_CATEGORICALS,
            reference=train_data
        )
        model = lgb.train(
            param_,
            train_data,
            50000,
            valid_sets=[train_data, valid_data],
            early_stopping_rounds=200,
            verbose_eval=200
        )
        importances = [("%s: %.2f" % x) for x in sorted(
            zip(feature_names, model.feature_importance("gain")),
            key=lambda x: x[1], reverse=True
        )]
        for name, val in zip(feature_names, model.feature_importance("gain")):
            global_importance[name] += val
        if verbose:
            print("-" * 20)
            print("\n".join(importances[:50]))
        print("=" * 20)
        with open(f"cache/importance_{i}.txt", "w") as fw:
            fw.write("\n".join(importances))
        test_preds.append(model.predict(
            df_test[feature_names], num_iteration=model.best_iteration
        ))
        val_pred = model.predict(
            df_train.iloc[val_index][feature_names],
            num_iteration=model.best_iteration
        )
        val_losses.append(
            mean_absolute_error(
                df_train.Next_Premium.iloc[val_index],
                val_pred
            )
        )
        val_pred_dfs.append(pd.DataFrame(
            {"simple_pred": val_pred}, index=val_index))
        print("Fold Val Loss: {:.2f}".format(val_losses[-1]))
    print("Val losses: {:.2f} +- {:.2f}".format(
        np.mean(val_losses), np.std(val_losses)))
    df_val_preds = pd.concat(val_pred_dfs, axis=0).sort_index()
    name = "lgb_simple_{}_{:.2f}".format(
        date.today().strftime("%m%d"), np.mean(val_losses)
    )
    df_val_preds.to_pickle(OUT_DIR / (name + ".pd"))
    joblib.dump(np.mean(test_preds, axis=0), OUT_DIR / (name + ".pkl"))
    df_val_preds.to_pickle("cache/simple_val.pd")
    joblib.dump(np.mean(test_preds, axis=0), "cache/simple_test.pkl")
    joblib.dump(global_importance, "cache/simple_importance.pkl")
    return np.mean(val_losses)


def main():
    df_train = pd.read_csv("data/training-set.csv")
    df_test = pd.read_csv("data/testing-set.csv").drop(
        "Next_Premium", axis=1)
    df_features = feature_engineering(df_train, df_test)
    # print("\nFeatures:")
    # print(df_features.sample(10))

    df_train = df_train.set_index("Policy_Number").join(df_features)
    df_test = df_test.set_index("Policy_Number").join(df_features)
    del df_train["index"]
    del df_test["index"]

    fit_and_predict(df_train, df_test)


if __name__ == "__main__":
    main()
