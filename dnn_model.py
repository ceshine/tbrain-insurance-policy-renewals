import gc
from datetime import datetime
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

np.set_printoptions(threshold=50, edgeitems=20)

AVERAGING_WINDOW = 300
SEED = 123
CHECKPOINT_DIR = Path("cache/model_cache/")
CHECKPOINT_DIR.mkdir(exist_ok=True)
LOG_DIR = Path("cache/logs/")
LOG_DIR.mkdir(exist_ok=True)
DEVICE = torch.device("cpu")


class CircularLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_mul, ratio, steps_per_cycle, decay=1, last_epoch=-1):
        self.max_mul = max_mul - 1
        self.turning_point = steps_per_cycle // (ratio + 1)
        self.steps_per_cycle = steps_per_cycle
        self.decay = decay
        self.history = []
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        residual = self.last_epoch % self.steps_per_cycle
        multiplier = self.decay ** (self.last_epoch // self.steps_per_cycle)
        if residual <= self.turning_point:
            multiplier *= self.max_mul * (residual / self.turning_point)
        else:
            multiplier *= self.max_mul * (
                (self.steps_per_cycle - residual) /
                (self.steps_per_cycle - self.turning_point))
        new_lr = [lr * (1 + multiplier) for lr in self.base_lrs]
        self.history.append(new_lr)
        return new_lr


class BaseBot:
    """Base Interface to Model Training and Inference"""

    name = "basebot"

    def __init__(self, model, train_loader, *, clip_grad=1, val_loader=None, avg_window=AVERAGING_WINDOW):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.avg_window = avg_window
        self.clip_grad = clip_grad
        self.model = model
        self.criterion = torch.nn.L1Loss()
        self.init_logging(LOG_DIR)
        self.best_performers = []
        self.base_steps = 0

    def init_logging(self, log_dir, debug=False):
        log_dir.mkdir(exist_ok=True)
        Path("cache/runs").mkdir(exist_ok=True)
        date_str = datetime.now().strftime('%Y-%m-%d_%H-%M')
        log_file = 'log_{}.txt'.format(date_str)
        formatter = logging.Formatter(
            '[[%(asctime)s]] %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p'
        )
        self.logger = logging.getLogger("bot")
        # Remove all existing handlers
        self.logger.handlers = []
        # Initialize handlers
        fh = logging.FileHandler(
            Path(log_dir) / Path(log_file))
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        self.logger.addHandler(sh)
        self.logger.setLevel(logging.INFO)
        if debug:
            self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        self.logger.info("SEED: %d", SEED)

    def save_state(self):
        torch.save(self.model.state_dict(), CHECKPOINT_DIR / "best.pth")

    def restore_state(self):
        self.model.load_state_dict(
            torch.load(CHECKPOINT_DIR / "best.pth"))

    def train(
            self, optimizer, n_iters, *, log_interval=50,
            early_stopping_cnt=0, scheduler=None,
            snapshot_interval=2500
    ):
        train_losses = deque(maxlen=self.avg_window)
        train_weights = deque(maxlen=self.avg_window)
        if self.val_loader is not None:
            best_val_loss = 10000
        step, epoch = 0, 0
        wo_improvement = 0
        self.best_performers = []
        self.logger.info("Optimizer {}".format(str(optimizer)))
        self.logger.info("Batches per epoch: {}".format(
            len(self.train_loader)))
        try:
            while step < n_iters:
                epoch += 1
                self.logger.info(
                    "=" * 20 + "Epoch {}".format(epoch) + "=" * 20)
                for *input_tensors, y in self.train_loader:
                    input_tensors = [x.to(DEVICE) for x in input_tensors]
                    self.model.train()
                    y = y.to(DEVICE)
                    assert self.model.training
                    optimizer.zero_grad()
                    output = self.model(*input_tensors)
                    batch_loss = self.criterion(output[:, 0], y)
                    batch_loss.backward()
                    train_losses.append(batch_loss.data.cpu().numpy())
                    train_weights.append(y.size(0))
                    clip_grad_norm_(self.model.parameters(), self.clip_grad)
                    optimizer.step()
                    step += 1
                    if (step % log_interval == 0 or
                            step % snapshot_interval == 0):
                        train_loss_avg = np.average(
                            train_losses, weights=train_weights)
                        self.logger.info("Step {}: train {:.6f} lr: {:.3e}".format(
                            step, train_loss_avg, optimizer.param_groups[0]['lr']))
                    if self.val_loader is not None and step % snapshot_interval == 0:
                        _, loss = self.predict(
                            self.val_loader, is_test=False)
                        loss_str = "%.6f" % loss
                        self.logger.info("Snapshot loss %s", loss_str)
                        target_path = (
                            CHECKPOINT_DIR /
                            "snapshot_{}_{}.pth".format(self.name, loss_str))
                        heapq.heappush(
                            self.best_performers, (loss, target_path))
                        torch.save(self.model.state_dict(), target_path)
                        self.logger.info(target_path)
                        assert Path(target_path).exists()
                        if best_val_loss > loss + 1e-4:
                            self.logger.info("New low\n")
                            # self.save_state()
                            best_val_loss = loss
                            wo_improvement = 0
                        else:
                            wo_improvement += 1
                    if scheduler:
                        # old_lr = optimizer.param_groups[0]['lr']
                        scheduler.step()
                        # if old_lr != optimizer.param_groups[0]['lr']:
                        #     # Reload best checkpoint
                        #     self.restore_state()
                    if (self.val_loader is not None and early_stopping_cnt and
                            wo_improvement > early_stopping_cnt):
                        return self.best_performers
                    if step >= n_iters:
                        break
        except KeyboardInterrupt:
            pass
        self.base_steps += step
        return self.best_performers

    def predict_avg(self, loader, k=8, *, is_test=False):
        preds = []
        self.logger.info("Predicting %s...", (
            "test" if is_test else "validation"))
        # Make a copy of the list
        best_performers = list(self.best_performers)
        for _ in range(k):
            target = heapq.heappop(best_performers)[1]
            self.logger.info("Loading %s", format(target))
            self.model.load_state_dict(torch.load(target))
            preds.append(self.predict(
                loader, is_test=is_test)[0].unsqueeze(0))
        return torch.cat(preds, dim=0).mean(dim=0)

    def predict(self, loader, *, is_test=False):
        self.model.eval()
        global_y = []
        outputs = []
        with torch.set_grad_enabled(False):
            for *input_tensors, y in loader:
                input_tensors = [x.to(DEVICE) for x in input_tensors]
                y = y.to(DEVICE)
                global_y.append(y)
                tmp = self.model(*input_tensors)
                outputs.append(tmp.data)
            res = torch.cat(outputs, dim=0)
        if is_test:
            return (res.data,)
        global_y = torch.cat(global_y, dim=0)
        loss = self.criterion(
            res[:, 0], global_y
        )
        return res.data, loss.cpu().data.numpy()


def get_model(num_features):
    model = nn.Sequential(
        weight_norm(nn.Linear(num_features, 1000)),
        nn.ELU(),
        nn.Dropout(0.5),
        weight_norm(nn.Linear(1000, 1000)),
        nn.ELU(),
        nn.Dropout(0.5),
        weight_norm(nn.Linear(1000, 1000)),
        nn.ELU(),
        nn.Dropout(0.5),
        weight_norm(nn.Linear(1000, 1))
    )
    for m in model:
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight_v)
            nn.init.kaiming_normal_(m.weight_g)
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
        cycle = batches_per_epoch * 20

        model = get_model(df_train.shape[1]-1)
        optimizer = torch.optim.Adam(
            model.parameters(), betas=(0.9, 0.999), lr=1e-3 / 128)
        scheduler = CircularLR(
            optimizer, max_mul=128, ratio=5,
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

        test_preds.append(bot.predict_avg(test_loader, k=3, is_test=True)[
            :, 0].cpu().numpy() * 1000)
        val_pred = bot.predict_avg(valid_loader, k=3, is_test=False)[
            :, 0].cpu().numpy()
        val_losses.append(
            mean_absolute_error(
                df_train.Next_Premium.iloc[val_index],
                val_pred
            )
        )
        val_pred_dfs.append(pd.DataFrame(
            {"dnn_pred": val_pred * 1000}, index=val_index))
        print("Fold Val Loss: {:.4f}".format(val_losses[-1]))
    print("Val losses: {:.4f} +- {:.4f}".format(
        np.mean(val_losses), np.std(val_losses)))
    df_val_preds = pd.concat(val_pred_dfs, axis=0).sort_index()
    df_val_preds.to_pickle("cache/dnn_val.pd")
    joblib.dump(np.mean(test_preds, axis=0), "cache/dnn_test.pkl")
    return np.mean(test_preds, axis=0), np.mean(val_losses)


def preprocess_features(df_train, df_test):
    categoricals = POLICY_FIXED_CATEGORICALS + EXTRA_CATEGORICALS
    numericals = list(set(df_train.columns) -
                      set(["Next_Premium"] + categoricals))
    # One-hot Encoding
    encoder = OneHotEncoder(min_obs=0)
    matrix = encoder.fit_transform(df_train[categoricals]).todense()
    onehot_categoricals = [f"categorical_{i}" for i in range(matrix.shape[1])]
    df_tmp = pd.DataFrame(
        matrix,
        columns=onehot_categoricals,
        index=df_train.index
    )
    df_train = pd.concat([
        df_train[numericals + ["Next_Premium"]],
        df_tmp], axis=1)
    matrix = encoder.transform(df_test[categoricals]).todense()
    df_tmp = pd.DataFrame(
        matrix,
        columns=onehot_categoricals,
        index=df_test.index
    )
    df_test = pd.concat([
        df_test[numericals + ["Next_Premium"]],
        df_tmp], axis=1)
    # Normalize
    scaler = StandardScaler(copy=True)
    scaler.fit(pd.concat([
        df_train[numericals + onehot_categoricals],
        df_test[numericals + onehot_categoricals]
    ], axis=0).values)
    df_train[numericals + onehot_categoricals] = scaler.transform(
        df_train[numericals + onehot_categoricals].values)
    df_test[numericals + onehot_categoricals] = scaler.transform(
        df_test[numericals + onehot_categoricals].values)
    df_train["Next_Premium"] = df_train["Next_Premium"] / 1000
    return df_train, df_test


def main():
    df_train = pd.read_csv("data/training-set.csv")
    df_test = pd.read_csv("data/testing-set.csv")
    df_features = feature_engineering(df_train, df_test)

    df_train = df_train.set_index("Policy_Number").join(df_features)
    df_test = df_test.set_index("Policy_Number").join(df_features)
    del df_features
    del df_train["index"]
    del df_test["index"]
    # del df_train["overlap_test"]
    # del df_test["overlap_test"]

    df_train, df_test = preprocess_features(df_train, df_test)
    fit_and_predict(df_train, df_test)


if __name__ == "__main__":
    main()
