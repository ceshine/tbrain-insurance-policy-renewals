from datetime import datetime
import pathlib

import pandas as pd
import numpy as np
import joblib

from utils import LabelEncoder, OneHotEncoder

POLICY_FIXED_CATEGORICALS = [
    'fsex', 'fmarriage',
    'Imported_or_Domestic_Car', 'Cancellation',
    'Vehicle_Make_and_Model1',
    'Distribution_Channel',
    'lia_class', 'plia_acc', 'pdmg_acc',
    'fassured',  "same_bdate",  # 'iply_area', 'aassured_zip',
    # 'fequipment1', 'fequipment2', 'fequipment3',
    # 'Multiple_Products_with_TmNewa_(Yes_or_No?)'
]

EXTRA_CATEGORICALS = [
]

MEMORY = joblib.Memory(cachedir="cache/")

SPLIT_POINT = 132854


@MEMORY.cache
def feature_engineering(df_train, df_test, data_path="data/"):
    df_claims = pd.read_csv(pathlib.Path(data_path) / "claim_0702.csv")
    df_policy = pd.read_csv(pathlib.Path(data_path) / "policy_0702.csv")

    # Fill NA
    df_policy["fsex"] = df_policy["fsex"].fillna(" ")
    df_policy["fmarriage"] = df_policy["fmarriage"].fillna(" ")

    # Index policies
    df_uniq_policy = df_policy[
        ["Policy_Number"]
    ].drop_duplicates()
    df_uniq_policy["index"] = range(df_uniq_policy.shape[0])
    df_uniq_policy.set_index("Policy_Number", inplace=True)
    # # Index Feature
    # df_uniq_policy["overlap_test"] = (
    #     df_uniq_policy["index"] >= SPLIT_POINT).astype("int8")
    # Index Order Features
    split_points = [-1, 95943, 132800, 189940, 252695, 294932, 351272]
    df_uniq_policy["relative_position"] = 0
    for i in range(1, len(split_points)):
        mask = (df_uniq_policy["index"] <= split_points[i]) & (
            df_uniq_policy["index"] > split_points[i-1])
        width = (split_points[i] - split_points[i-1] - 1)
        df_uniq_policy.loc[mask, "relative_position"] = (
            (df_uniq_policy.loc[mask, "index"] - split_points[i-1] - 1) / width)
    df_uniq_policy.to_pickle("cache/uniq_policy.pd")

    # Process claims
    coverage_mappings = {
        k: v for _, (v, k) in df_policy[[
            "Main_Insurance_Coverage_Group", "Insurance_Coverage"
        ]].drop_duplicates().iterrows()
    }
    df_claims["main_coverage_group"] = df_claims['Coverage'].apply(
        lambda x: coverage_mappings[x])
    # At Fault
    df_claims.loc[df_claims['At_Fault?'] > 100, 'At_Fault?'] = 100
    df_claims.loc[(df_claims['At_Fault?'] > 0) & (
        df_claims['At_Fault?'] < 100), 'At_Fault?'] = 50
    # TODO: CHECK NA
    df_claims['At_Fault?'] = df_claims['At_Fault?'].fillna(
        0).astype("int32").astype("category")
    # Clean Coverages
    encoder = LabelEncoder(min_obs=1000)
    tmp = encoder.fit_transform(df_claims[["Coverage"]])
    df_claims["Coverage"] = tmp["Coverage"].astype("int32")
    # Payments
    df_claims["Total_Paid"] = (
        df_claims["Paid_Loss_Amount"] +
        df_claims["paid_Expenses_Amount"])
    df_claims["Total_Paid_at_Fault"] = (
        df_claims["At_Fault?"].values * df_claims["Total_Paid"].values)
    df_claims_per_policy = df_uniq_policy[["index"]].join(
        df_claims.groupby("Policy_Number")[[
            "Total_Paid",  "Total_Paid_at_Fault", "Deductible"
        ]].sum(), how="right"
    ).fillna(0).drop("index", axis=1)
    df_claims_per_policy = df_claims_per_policy.join(
        df_claims.groupby("Policy_Number")[["Claim_Number"]].nunique(),
        how="left"
    ).fillna(0)
    df_claims_per_policy.rename(
        columns={
            "Claim_Number": "claims",
            "Total_Paid": "claim_total_paid",
            "Total_Paid_at_Fault": "claim_total_paid_at_fault",
            "Deductible": "claim_total_deductible"
        }, inplace=True)
    df_claims_per_policy = df_claims_per_policy.join(
        df_claims.groupby("Policy_Number")[
            "Claim_Number"].size().to_frame("claim_entries"),
        how="left"
    ).fillna(0)
    # Per Coverage Group
    df_claims_main_coverage = df_claims.groupby(
        ["Policy_Number", "main_coverage_group"]
    )[["Total_Paid", "Deductible"]].sum().unstack(-1, fill_value=0)
    df_claims_main_coverage.columns = [
        "claims_%s_%s" % (x, y) for x, y in zip(
            df_claims_main_coverage.columns.get_level_values(1),
            df_claims_main_coverage.columns.get_level_values(0)
        )
    ]
    df_claims_per_policy = df_claims_per_policy.join(
        df_claims_main_coverage, how="left")
    # Relationship with Insured
    df_claims_relations = df_claims[[
        "Policy_Number", "Claim_Number", "Driver's_Relationship_with_Insured"
    ]].drop_duplicates().groupby(
        ["Policy_Number", "Driver's_Relationship_with_Insured"]
    )["Claim_Number"].size().unstack(-1, fill_value=0)
    df_claims_relations.columns = [
        "relation_%s" % x for x in df_claims_relations.columns
    ]
    df_claims_per_policy = df_claims_per_policy.join(
        df_claims_relations, how="left")
    # # Per Coverage
    # df_claims_coverage = df_claims.groupby(
    #     ["Policy_Number", "Coverage"]
    # )["Total_Paid"].agg(["count", "sum"]).unstack(-1, fill_value=0)
    # df_claims_coverage.columns = [
    #     "claims_coverage_%s_%s" % (x, y) for x, y in zip(
    #         df_claims_coverage.columns.get_level_values(1),
    #         df_claims_coverage.columns.get_level_values(0)
    #     )
    # ]
    # df_claims_per_policy = df_claims_per_policy.join(
    #     df_claims_coverage, how="left")
    # At Fault
    # df_claims_at_fault = df_claims[[
    #     "Policy_Number", "Claim_Number", "At_Fault?", "Total_Paid"
    # ]].drop_duplicates().groupby(
    #     ["Policy_Number", "At_Fault?"]
    # )["Total_Paid"].agg(["sum", "count"]).unstack(-1, fill_value=0)  # .unstack(-1, fill_value=0)
    # df_claims_at_fault.columns = [
    #     "claims_at_fault_%s_%s" % (x, y) for x, y in zip(
    #         df_claims_at_fault.columns.get_level_values(0),
    #         df_claims_at_fault.columns.get_level_values(1)
    #     )
    # ]
    # df_claims_per_policy = df_claims_per_policy.join(
    #     df_claims_at_fault, how="left")
    # df_claims_per_policy["claim_total_paid"] = np.log1p(
    #     df_claims_per_policy["claim_total_paid"])
    # df_claims_per_policy["claim_total_deductible"] = np.log1p(
    #     df_claims_per_policy["claim_total_deductible"])
    df_claims_per_policy["claims_車責_Total_Paid"] = np.log1p(
        df_claims_per_policy["claims_車責_Total_Paid"])
    df_claims_per_policy["claims_車損_Total_Paid"] = np.log1p(
        df_claims_per_policy["claims_車損_Total_Paid"])
    del df_claims_per_policy["claims_竊盜_Total_Paid"]

    print("\nClaims per policy")
    print(df_claims_per_policy.sample(10))

    # Claim Date Statistics
    df_claims["Date"] = df_claims["Accident_Date"].apply(
        lambda x: datetime.strptime(x, "%Y/%m"))
    df_claim_dates = df_claims.groupby("Policy_Number")[
        "Date"].min().to_frame("min_date")
    # df_claim_dates["min_month"] = df_claim_dates["min_date"].dt.month
    # df_claim_dates["min_year"] = df_claim_dates["min_date"].dt.year
    for col in ["min_date"]:
        df_claim_dates[col] = df_claim_dates[col].astype("int64")
        # df_claim_dates[col] = df_claim_dates[col].astype(
        #     "int64") / 10**9 / 60 / 60 / 24  # ns / s / m / d
    # fill policy
    df_claim_dates = df_claim_dates.join(
        df_uniq_policy[["index"]], how="right").sort_values("index").drop("index", axis=1)
    # df_claim_dates["min_month"] = df_claim_dates["min_month"].fillna(
    #     method="ffill").fillna(method="bfill")
    # df_claim_dates["min_year"].fillna(-1, inplace=True)
    # df_claim_dates["min_date"] = df_claim_dates["min_date"].fillna(
    #     method="ffill").fillna(method="bfill")
    df_claim_dates["min_date_smooth_1000"] = df_claim_dates["min_date"].rolling(
        1000, min_periods=20, center=True
    ).mean().fillna(method="bfill").fillna(method="ffill") / 10**9 / 60 / 60 / 24
    df_claim_dates["min_date_dayofyear"] = pd.to_datetime(df_claim_dates["min_date"].rolling(
        1000, min_periods=20, center=True
    ).min().fillna(method="bfill").fillna(method="ffill")).dt.dayofyear
    # weights = np.concatenate(
    #     [np.arange(1, 501), np.arange(500, 0, -1)], axis=0)
    # weights = weights / np.sum(weights)
    # def f(x):
    #     return np.sum(weights * x)
    # df_claim_dates["min_date_weighted_2000"] = df_claim_dates["min_date"].rolling(
    #     2000, min_periods=20, center=False, win_type="blackman"
    # ).mean().fillna(method="bfill").fillna(method="ffill")
    # df_claim_dates["min_date_ewm_500"] = df_claim_dates["min_date"].ewm(
    #     halflife=500, min_periods=1
    # ).mean().fillna(method="ffill").fillna(method="bfill")
    del df_claim_dates["min_date"]
    # df_claim_dates.rename(columns={
    #     "min": "min_claim_date", "max": "max_claim_date"}, inplace=True)
    print("\n Claim Dates")
    print(df_claim_dates.sample(10))

    # Policy Premium Stats
    # df_policy["Premium_log"] = np.log1p(df_policy["Premium"])
    df_policy_premiums = df_policy.groupby("Policy_Number")["Premium"].agg(
        ["count", "min", "max", "sum"])
    df_policy_premiums.rename(
        columns={
            "count": "n_coverages", "sum": "total_premium",
            "min": "min_premium", "max": "max_premium"
        }, inplace=True
    )
    print("\nPolicy_Premium:")
    print(df_policy_premiums.sample(10))

    # Polic Premium Stats by Main Coverage
    df_policy_main_premiums = df_policy.groupby(
        ["Policy_Number", "Main_Insurance_Coverage_Group"]
    )["Premium"].agg(["count", "sum"]).unstack(-1, fill_value=0)
    df_policy_main_premiums.columns = [
        "premium_%s_%s" % (x, y) for x, y in zip(
            df_policy_main_premiums.columns.get_level_values(1),
            df_policy_main_premiums.columns.get_level_values(0)
        )
    ]
    print("\nPolicy_Main_Premium:")
    print(df_policy_main_premiums.sample(10))

    # Clean Coverages
    encoder = LabelEncoder(min_obs=5000)
    tmp = encoder.fit_transform(df_policy[["Insurance_Coverage"]])
    df_policy["Insurance_Coverage"] = tmp["Insurance_Coverage"].astype(
        "int32")

    # Policy Premium Stats by Coverage
    df_policy_coverage_premiums = df_policy.groupby(
        ["Policy_Number", "Insurance_Coverage"]
    )["Premium"].agg(["count", "sum"]).unstack(-1, fill_value=0)
    df_policy_coverage_premiums.columns = [
        "premium_coverage_%s_%s" % (x, y) for x, y in zip(
            df_policy_coverage_premiums.columns.get_level_values(1),
            df_policy_coverage_premiums.columns.get_level_values(0)
        )
    ]
    print("\nPolicy_Coverage_Premium:")
    print(df_policy_coverage_premiums.sample(10))

    # Other Policy Aggs
    df_policy["Total_Insured_Amount"] = (
        df_policy["Insured_Amount1"] + df_policy["Insured_Amount2"] +
        df_policy["Insured_Amount3"]
    )
    df_policy_aggs = df_policy.groupby("Policy_Number")[[
        # , "Coverage_Deductible_if_applied"
        "Total_Insured_Amount", "Insured_Amount1", "Insured_Amount2", "Insured_Amount3",
    ]].sum()
    print("\nPolicy Aggs:")
    print(df_policy_aggs.sample(10))

    # Other Policy Aggs By Main Coverage
    df_policy_main_aggs = df_policy.groupby(["Policy_Number", "Main_Insurance_Coverage_Group"])[[
        # "Coverage_Deductible_if_applied"
        "Insured_Amount1", "Insured_Amount2", "Insured_Amount3", "Total_Insured_Amount"
    ]].sum().unstack(-1, fill_value=0)
    df_policy_main_aggs.columns = [
        "%s_%s" % (x, y) for x, y in zip(
            df_policy_main_aggs.columns.get_level_values(1),
            df_policy_main_aggs.columns.get_level_values(0)
        )
    ]
    print("\nPolicy Aggs by Main:")
    print(df_policy_main_aggs.sample(10))

    # Encode Categoricals
    encoder = LabelEncoder(min_obs=7500)
    df_policy["same_bdate"] = (df_policy["ibirth"] == df_policy["dbirth"]) * 1
    df_policy_fixed_categoricals = df_policy[
        ["Policy_Number"] + POLICY_FIXED_CATEGORICALS
    ].drop_duplicates().set_index("Policy_Number")
    df_policy_fixed_categoricals = encoder.fit_transform(
        df_policy_fixed_categoricals[POLICY_FIXED_CATEGORICALS]
    )
    print("\nPolicy-fixed Categoricals")
    print(df_policy_fixed_categoricals.nunique())

    # Policy-fixed Numeric Categoricals
    df_policy_fixed_numericals = df_policy[[
        "Policy_Number", "Replacement_cost_of_insured_vehicle", "ibirth", "dbirth",
        "Engine_Displacement_(Cubic_Centimeter)", 'Manafactured_Year_and_Month'
    ]].drop_duplicates().set_index("Policy_Number")
    # df_policy_fixed_numericals["differnt_birth"] = (
    #     df_policy_fixed_numericals["ibirth"] != df_policy_fixed_numericals["dbirth"])
    # del df_policy_fixed_numericals["dbirth"]
    df_policy_fixed_numericals['ibirth'] = df_policy_fixed_numericals['ibirth'].str.extract(
        '(19..)', expand=True).fillna(value=1968).astype("int32")
    df_policy_fixed_numericals['dbirth'] = df_policy_fixed_numericals['dbirth'].str.extract(
        '(19..)', expand=True).fillna(value=1968).astype("int32")

    # Prior Policy
    df_prior_policy = df_policy[[
        "Policy_Number", "Prior_Policy_Number"
    ]].drop_duplicates().fillna("New")
    # df_prior_policy["first_time_policy"] = df_prior_policy["Prior_Policy_Number"] == "New"
    df_prior_policy = df_prior_policy.merge(
        df_policy_premiums[["total_premium", "n_coverages"]].reset_index(),
        left_on="Prior_Policy_Number", right_on="Policy_Number", how="left",
        suffixes=["", "_prev"]
    ).set_index("Policy_Number").fillna(0)
    del df_prior_policy["Prior_Policy_Number"]
    del df_prior_policy["Policy_Number_prev"]
    df_prior_policy.rename(columns={
        "total_premium": "prev_total_premium",
        "n_coverages": "n_prev_coverages"
    }, inplace=True)

    # Coverage
    encoder = OneHotEncoder(min_obs=10000)
    df_coverage = df_policy[["Policy_Number"]].copy()
    sparse = encoder.fit_transform(
        df_policy[["Insurance_Coverage"]])
    column_names = [f"coverage_{i}" for i in range(sparse.shape[1])]
    df_coverage = pd.concat([
        df_coverage, pd.DataFrame(sparse.todense(), columns=column_names)
    ], axis=1).reset_index(drop=True)
    df_coverage = df_coverage.groupby("Policy_Number").sum()

    df_features = df_policy_premiums.join(
        df_claims_per_policy, how="left"
    ).fillna(0).join(
        df_policy_main_premiums
    ).fillna(0).join(
        df_policy_coverage_premiums
    ).join(
        df_policy_aggs
    ).join(
        df_policy_main_aggs
    ).join(
        df_policy_fixed_categoricals
    ).join(
        df_policy_fixed_numericals
    ).join(
        df_prior_policy
    ).join(
        df_coverage
    ).join(
        df_uniq_policy
    ).join(
        df_claim_dates
    )

    # Meta Features
    df_features["premium_paid_ratio"] = np.nan_to_num(
        df_features["claim_total_paid"] / df_features["total_premium"])
    df_features["premium_paid_at_fault_ratio"] = np.nan_to_num(
        df_features["claim_total_paid_at_fault"] / df_features["total_premium"])
    assert df_policy_premiums.shape[0] == df_features.shape[0]
    print("Feature Shape:", df_features.shape)
    return df_features
