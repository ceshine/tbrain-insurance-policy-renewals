import sys
import glob

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error
MODEL_PATH = "cache/ens_2/"


def main():
    model_files = [
        ".".join(x.split(".")[:-1])
        for x in glob.glob(MODEL_PATH) if x.endswith(".pd")]

    val_tmp, test_tmp = [], []
    y = pd.read_csv("data/training-set.csv")["Next_Premium"].values
    df_test = pd.read_csv("data/testing-set.csv")[["Policy_Number"]]
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
    print("=" * 20)
    print(np.stack(val_tmp, axis=1).shape)
    print(np.corrcoef(np.stack(val_tmp, axis=1), rowvar=False))

    print("=" * 20)
    print("Test")
    for filename in sys.argv[1:]:
        test_tmp.append(
            np.clip(joblib.load(MODEL_PATH + filename + ".pkl"), 0, 2e8)
        )
        print("%.2f %.2f %.2f %.2f %.2f" % (
            np.min(test_tmp), np.percentile(test_tmp, 25),
            np.median(test_tmp), np.percentile(test_tmp, 75),
            np.max(test_tmp)))
    print("=" * 20)
    print(np.stack(test_tmp, axis=1).shape)
    print(np.corrcoef(np.stack(test_tmp, axis=1), rowvar=False))

    val_preds = np.mean(val_tmp, axis=0)
    print("Val loss: %.2f" % mean_absolute_error(y, val_preds))
    test_preds = np.mean(test_tmp, axis=0)
    df_test["Next_Premium"] = test_preds
    print(df_test.head())
    df_test.to_csv("sub_ens.csv", index=False, float_format="%.4f")


if __name__ == "__main__":
    main()
