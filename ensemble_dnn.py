import glob
import sys
import os
import random
from datetime import datetime, date
from collections import deque
import heapq
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.nn.utils.weight_norm import weight_norm

from features import feature_engineering, POLICY_FIXED_CATEGORICALS, EXTRA_CATEGORICALS
from config import KFOLD_SEED, KFOLD_N
from utils import OneHotEncoder
from dnn_model import BaseBot, CircularLR, preprocess_features, DEVICE

np.set_printoptions(threshold=50, edgeitems=20)

AVERAGING_WINDOW = 300
SEED = int(os.environ.get("SEED", 123))
CHECKPOINT_DIR = Path("cache/model_cache/")
CHECKPOINT_DIR.mkdir(exist_ok=True)
LOG_DIR = Path("cache/logs/")
LOG_DIR.mkdir(exist_ok=True)
OUT_DIR = Path("cache/ens/")
OUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = "cache/single/*"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)


def get_model(num_features):
    model = nn.Sequential(
        weight_norm(nn.Linear(num_features, 1000)),
        # nn.Linear(num_features, 1000),
        nn.ReLU(),
        #  nn.BatchNorm1d(1000),
        nn.Dropout(0.2),
        # nn.Linear(1000, 1000),
        # nn.ReLU(),
        # nn.BatchNorm1d(1000),
        # nn.Dropout(0.3),
        # weight_norm(nn.Linear(1000, 1000)),
        # nn.ReLU(),
        # nn.Dropout(0.2),
        weight_norm(nn.Linear(1000, 1))
    )
    # model = nn.Sequential(
    #     nn.Linear(num_features, 1000),
    #     nn.ELU(),
    #     nn.LayerNorm(1000),
    #     nn.Dropout(0.2),
    #     nn.Linear(1000, 1000),
    #     nn.ELU(),
    #     nn.LayerNorm(1000),
    #     nn.Dropout(0.2),
    #     nn.Linear(1000, 1)
    # )
    for m in model:
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight_v)
            nn.init.kaiming_normal_(m.weight_g)
            # nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
    return model.to(DEVICE)


def fit_and_predict(df_train, df_test):
    batch_size = 32
    kf = KFold(n_splits=KFOLD_N, random_state=KFOLD_SEED, shuffle=True)
    val_losses = []
    val_pred_dfs = []
    test_preds = []
    test_dataset = TensorDataset(
        torch.from_numpy(df_test.drop("Next_Premium", axis=1).values).float(),
        torch.from_numpy(df_test.Next_Premium.values).float()
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size*4, shuffle=False, num_workers=0
    )
    for i, (train_index, val_index) in enumerate(kf.split(df_train)):
        print("-" * 20)
        print(f"Fold {i+1}")
        print("-" * 20)
        batches_per_epoch = len(train_index) // batch_size
        print("Batches per epoch: ", batches_per_epoch)
        cycle = batches_per_epoch * 10

        model = get_model(df_train.shape[1]-1)
        optimizer = torch.optim.Adam(
            model.parameters(), betas=(0.9, 0.999), lr=5e-4 / 64)
        scheduler = CircularLR(
            optimizer, max_mul=64, ratio=5,
            steps_per_cycle=cycle
        )

        train_dataset = TensorDataset(
            torch.from_numpy(df_train.iloc[train_index].drop(
                "Next_Premium", axis=1).values).float(),
            torch.from_numpy(
                df_train.iloc[train_index].Next_Premium.values).float()
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
        valid_dataset = TensorDataset(
            torch.from_numpy(df_train.iloc[val_index].drop(
                "Next_Premium", axis=1).values).float(),
            torch.from_numpy(
                df_train.iloc[val_index].Next_Premium.values).float()
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=batch_size*4, shuffle=False, num_workers=0)

        bot = BaseBot(
            model, train_loader, clip_grad=5.,
            val_loader=valid_loader, avg_window=500)

        bot.train(
            optimizer, n_iters=cycle,
            log_interval=batches_per_epoch // 10,
            snapshot_interval=batches_per_epoch // 10 * 5,
            early_stopping_cnt=15, scheduler=scheduler)

        test_preds.append(bot.predict_avg(test_loader, k=2, is_test=True)[
            :, 0].cpu().numpy() * 1000)
        val_pred = bot.predict_avg(valid_loader, k=2, is_test=False)[
            :, 0].cpu().numpy()
        val_losses.append(
            mean_absolute_error(
                df_train.Next_Premium.iloc[val_index],
                val_pred
            ) * 1000
        )
        val_pred_dfs.append(pd.DataFrame(
            {"dnn_pred": val_pred * 1000}, index=val_index))
        print("Fold Val Loss: {:.4f}".format(val_losses[-1]))
    print("Val losses: {:.4f} +- {:.4f}".format(
        np.mean(val_losses), np.std(val_losses)))
    df_val_preds = pd.concat(val_pred_dfs, axis=0).sort_index()
    name = "ens_dnn_{}_{:.0f}".format(
        date.today().strftime("%m%d"), np.mean(val_losses) * 100
    )
    df_val_preds.to_pickle(OUT_DIR / (name + ".pd"))
    joblib.dump(np.mean(test_preds, axis=0), OUT_DIR / (name + ".pkl"))
    return np.mean(test_preds, axis=0), np.mean(val_losses)


def main():
    df_train = pd.read_csv("data/training-set.csv")
    df_test = pd.read_csv("data/testing-set.csv")
    df_features = feature_engineering(df_train, df_test)

    model_files = [
        ".".join(x.split(".")[:-1])
        for x in glob.glob(MODEL_PATH) if x.endswith(".pd")]
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
    df_train, df_test = preprocess_features(df_train, df_test)

    df_val_ens.set_index(df_train.index, inplace=True)
    df_test_ens.set_index(df_test.index, inplace=True)
    df_train = pd.concat([df_train, df_val_ens], axis=1)
    df_test = pd.concat([df_test, df_test_ens], axis=1)

    scaler = StandardScaler(copy=True)
    columns = model_files
    scaler.fit(pd.concat([
        df_test_ens[columns], df_val_ens[columns]
    ], axis=0).values)
    df_train[columns] = scaler.transform(
        df_train[columns].values)
    df_test[columns] = scaler.transform(
        df_test[columns].values)
    print("train:\n", df_train[columns].describe())
    print("test:\n", df_test[columns].describe())

    df_test["Next_Premium"], loss = fit_and_predict(
        df_train[columns + ["Next_Premium"]], df_test[columns + ["Next_Premium"]])
    df_test["Next_Premium"] = np.clip(df_test["Next_Premium"], 0, 5e5)
    df_test[["Next_Premium"]].to_csv(
        "sub_ens_dnn_{}_{:.0f}.csv".format(
            date.today().strftime("%m%d"), loss * 10000
        ), float_format="%.2f")


if __name__ == "__main__":
    main()
