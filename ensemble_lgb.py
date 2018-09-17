import os
import sys
import glob

from datetime import date
from pathlib import Path

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

from features import feature_engineering, POLICY_FIXED_CATEGORICALS, EXTRA_CATEGORICALS
from config import KFOLD_SEED, KFOLD_N

MEMORY = joblib.Memory(cachedir="cache/")
OUT_DIR = Path("cache/ens/")
OUT_DIR.mkdir(exist_ok=True)
SEED = int(os.environ.get("SEED", 123))

MODEL_PATH = "cache/single/*"


def fit_and_predict(df_train, df_test, params={}, verbose=True, n_best_features=None, ens_features=[]):
    kf = KFold(n_splits=KFOLD_N, random_state=KFOLD_SEED, shuffle=True)
    param_ = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression_l1',
        'feature_fraction': 0.8,
        'bagging_fraction': 0.85,
        'learning_rate': 0.02,
        'bagging_freq': 1,
        'min_data_in_leaf': 50,
        'max_depth': 9,
        'num_threads': 8,
        'seed': SEED,
        'num_leaves': 8,
        'cat_l2': 10,
        'lambda_l1': 0,
        'lambda_l2': 0
    }
    param_.update(params)
    val_losses = []
    if n_best_features is not None:
        importances = sorted(
            joblib.load("cache/simple_importance.pkl").items(),
            key=lambda x: x[1], reverse=True
        )
        feature_names = [
            x[0] for x in importances[:n_best_features]] + ens_features
    else:
        feature_names = df_train.drop(
            "Next_Premium", axis=1).columns.tolist()
    categorical_features = list(set(
        POLICY_FIXED_CATEGORICALS + EXTRA_CATEGORICALS).intersection(set(feature_names)))
    val_pred_dfs = []
    test_preds = []
    for i, (train_index, val_index) in enumerate(kf.split(df_train)):
        print("-" * 20)
        print(f"Fold {i+1}")
        print("-" * 20)
        train_data = lgb.Dataset(
            df_train.iloc[train_index][feature_names],
            label=df_train.Next_Premium.iloc[train_index],
            categorical_feature=categorical_features
        )
        valid_data = lgb.Dataset(
            df_train.iloc[val_index][feature_names],
            label=df_train.Next_Premium.iloc[val_index],
            categorical_feature=categorical_features,
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
            {"ens_pred": val_pred}, index=val_index))
        print("Fold Val Loss: {:.4f}".format(val_losses[-1]))
    print("Val losses: {:.4f} +- {:.4f}".format(
        np.mean(val_losses), np.std(val_losses)))
    df_val_preds = pd.concat(val_pred_dfs, axis=0).sort_index()
    name = "ens_lgb_{}_{:.0f}".format(
        date.today().strftime("%m%d"), np.mean(val_losses) * 100
    )
    df_val_preds.to_pickle(OUT_DIR / (name + ".pd"))
    joblib.dump(np.mean(test_preds, axis=0), OUT_DIR / (name + ".pkl"))
    return np.mean(test_preds, axis=0), np.mean(val_losses)


def main():
    df_train = pd.read_csv("data/training-set.csv")
    df_test = pd.read_csv("data/testing-set.csv").drop(
        "Next_Premium", axis=1)
    df_features = feature_engineering(df_train, df_test)
    # print("\nFeatures:")
    # print(df_features.sample(10))

    model_files = [
        ".".join(x.split(".")[:-1])
        for x in glob.glob(MODEL_PATH) if x.endswith(".pd")]
    print(model_files)
    val_tmp, test_tmp = [], []
    print("Validation")
    for filename in model_files:
        val_tmp.append(
            np.clip(pd.read_pickle(
                filename + ".pd"
            ).values[:, 0], 0, 2e8)
        )
        print("%.2f %.2f %.2f %.2f %.2f" % (
            np.min(val_tmp), np.percentile(val_tmp, 25),
            np.median(val_tmp), np.percentile(val_tmp, 75),
            np.max(val_tmp)))
    df_val_ens = pd.DataFrame(np.stack(val_tmp, axis=1), columns=model_files)
    # print(df_val_ens.head())

    print("=" * 20)
    print("Test")
    for filename in model_files:
        test_tmp.append(
            np.clip(joblib.load(filename + ".pkl"), 0, 2e8)
        )
        print("%.2f %.2f %.2f %.2f %.2f" % (
            np.min(test_tmp), np.percentile(test_tmp, 25),
            np.median(test_tmp), np.percentile(test_tmp, 75),
            np.max(test_tmp)))
    df_test_ens = pd.DataFrame(
        np.stack(test_tmp, axis=1), columns=model_files)
    print("=" * 20)
    # print(df_test_ens.head())

    df_train = df_train.set_index("Policy_Number").join(df_features)
    df_test = df_test.set_index("Policy_Number").join(df_features)
    df_val_ens.set_index(df_train.index, inplace=True)
    df_test_ens.set_index(df_test.index, inplace=True)
    df_train = pd.concat([df_train, df_val_ens], axis=1, ignore_index=False)
    df_test = pd.concat([df_test, df_test_ens], axis=1, ignore_index=False)
    print(df_train.head())
    # df_train["ratio_pred_nom"] = np.clip(
    #     df_train["ratio_pred"], 0, 2) * df_train["total_premium"]
    # df_test["ratio_pred_nom"] = np.clip(
    #     df_test["ratio_pred"], 0, 2) * df_test["total_premium"]
    del df_train["index"]
    del df_test["index"]

    df_test["Next_Premium"], loss = fit_and_predict(
        df_train, df_test, n_best_features=int(sys.argv[1]),
        ens_features=model_files)
    df_test["Next_Premium"] = np.clip(df_test["Next_Premium"], 0, 5e5)
    df_test[["Next_Premium"]].to_csv(
        "sub_ens_{}_{:.0f}.csv".format(
            date.today().strftime("%m%d"), loss * 100
        ), float_format="%.2f")


if __name__ == "__main__":
    main()
