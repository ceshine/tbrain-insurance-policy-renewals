
# coding: utf-8

# Because this notebook is quite large, enabling the `collapsible_headings` extension is recommended.

# In[1]:


get_ipython().magic('load_ext watermark')
get_ipython().magic('watermark -m -p pandas,numpy,matplotlib,seaborn')


# In[2]:


from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils import LabelEncoder

pd.set_option('display.float_format', lambda x: '%.2f' % x)
get_ipython().magic('matplotlib inline')


# ## Read Data 

# In[3]:


df_claims = pd.read_csv("../data/claim_0702.csv")
df_policy = pd.read_csv("../data/policy_0702.csv")
df_train  = pd.read_csv("../data/training-set.csv")
df_test   = pd.read_csv("../data/testing-set.csv")


# In[4]:


# Take a Peek
df_train.head()


# In[5]:


# Check Duplicates
assert df_train.Policy_Number.unique().shape[0] == df_train.shape[0]


# In[6]:


# Take a Peek
df_test.head()


# In[7]:


# Check Duplicates
assert df_test.Policy_Number.unique().shape[0] == df_test.shape[0]


# In[8]:


# Take a Peek
df_claims.head()


# In[9]:


# Take a Peek
df_policy.head()


# ## Examine Data

# ### Check Policies

# #### Overview
# Same policy can have multiple rows in df_policy. Here's an example that can help us find which columns can be different within the same policy:

# In[10]:


df_policy.iloc[0] == df_policy.iloc[1]


# Take a look at the actual data values:

# In[11]:


df_policy.iloc[:2].transpose()


# Another example:

# In[12]:


df_policy.iloc[2:4].transpose()


# The Number of Policies and the Number of Rows in df_policy (from these you get the average rows per policy):

# In[13]:


df_policy.shape[0]


# In[14]:


df_policy.Policy_Number.unique().shape[0]


# Check the statistics of the premiums from the same policy:

# In[15]:


df_policy_agg = df_policy.groupby("Policy_Number")["Premium"].agg(["count", "min", "max", "sum"]).reset_index()
df_policy_agg.describe()


# Summary:

# In[16]:


print("# of policies: ", df_policy.shape[0])
print("# of unique policies: ", df_policy_agg.shape[0])
print("# of claims: ", df_claims.shape[0])
print("# of train policies: ", df_train.shape[0])
print("# of test policies: ", df_test.shape[0])


# #### Relationship between the current premium and the next premium

# In[17]:


# Crafting "Total_Paid" column
df_claims["Total_Paid"] = df_claims["Paid_Loss_Amount"] + df_claims["paid_Expenses_Amount"]
# Group claims by policy
claims_per_policy = pd.concat(
    [df_train, df_test], axis=0)[["Policy_Number"]].merge(
    df_claims.groupby("Policy_Number")["Paid_Loss_Amount"].agg(["count", "sum"]).reset_index(), 
    on="Policy_Number", how="left"
).fillna(0)
claims_per_policy.rename(columns={"count": "claims", "sum": "Total_Paid"}, inplace=True)


# In[18]:


df_train_agg = df_policy_agg.merge(df_train, on="Policy_Number")
df_train_agg = df_train_agg.merge(claims_per_policy, on="Policy_Number")
df_train_agg.head()


# In[19]:


fig = plt.figure(figsize=(10, 5))
ax = sns.distplot(
    np.log1p(df_train_agg["Next_Premium"]), bins=100, color="skyblue",
    kde=False
)
ax.set_title("Distribution of Next_Premium (Log Scale)")


# In[20]:


nominal_changes = (df_train_agg["Next_Premium"] - df_train_agg["sum"])
nominal_changes.describe()


# In[21]:


fig = plt.figure(figsize=(10, 5))
ax = sns.distplot(
    nominal_changes[df_train_agg.Next_Premium > 0].values, bins=100, color="skyblue",
    kde=False
)
ax.set_title("Distribution of Nominal Changes (excluding Next_Premium == 0)")


# In[22]:


percentage_changes = nominal_changes / df_train_agg["sum"] * 100
percentage_changes.describe()


# In[23]:


# Outliers
percentage_changes[percentage_changes > 500].shape


# In[24]:


fig = plt.figure(figsize=(10, 5))
ax = sns.distplot(
    percentage_changes[(df_train_agg.Next_Premium > 0) & (percentage_changes<200)].values, 
    bins=100, color="skyblue",
    kde=False
)
ax.set_title("Distribution of Percentage Changes(excluding Next_Premium == 0 and outliers)")


# Plot Next_Premium against Sum of Policy Premiums (Excluding Next_Premium == 0. In log scale.):

# In[25]:


fig = plt.figure(figsize=(8, 8))
sample = df_train_agg[df_train_agg.Next_Premium != 0].sample(500)
ax = sns.regplot(
    x=np.log1p(sample["sum"]), 
    y=np.log1p(sample["Next_Premium"]), 
    scatter_kws={'alpha': 0.3})


# How about the minimums of policy premiums? (in log scale) :

# In[26]:


fig = plt.figure(figsize=(8, 8))
sample = df_train_agg[df_train_agg.Next_Premium != 0].sample(500)
ax = sns.regplot(
    x=np.log1p(sample["min"]), 
    y=np.log1p(sample["Next_Premium"]), 
    scatter_kws={'alpha': 0.3})


# And how about the maximums? (in log scale):

# In[27]:


fig = plt.figure(figsize=(8, 8))
sample = df_train_agg[df_train_agg.Next_Premium != 0].sample(500)
ax = sns.regplot(
    x=np.log1p(sample["max"]), 
    y=np.log1p(sample["Next_Premium"]), 
    scatter_kws={'alpha': 0.3})


# Those which changed the most (nominally):

# In[28]:


tmp = df_train_agg.iloc[np.argsort(np.abs(nominal_changes))[-100:][::-1]]
tmp[tmp.Next_Premium > 0][:10]


# Those which changed the most (in percentage):

# In[29]:


tmp = df_train_agg.iloc[np.argsort(np.abs(percentage_changes))[-100:][::-1]]
tmp[tmp.Next_Premium > 0][:10]


# Those which did not change at all:

# In[30]:


tmp = df_train_agg[df_train_agg.Next_Premium == df_train_agg["sum"]]
print(tmp.shape[0], tmp.shape[0] / df_train_agg.shape[0] * 100)
tmp.head()


# #### Find columns that could change within a policy

# Columns with max > 1 are the suspects:

# In[31]:


df_policy.groupby("Policy_Number")[[
    'Insured\'s_ID', 'Prior_Policy_Number', 'Vehicle_identifier', 
    'Main_Insurance_Coverage_Group', 'Insurance_Coverage',
    'Premium', 'ibirth', 'dbirth',
    'fequipment1', 'fequipment2', 'fequipment3',
]].nunique().describe()


# Some other random checkings:

# In[32]:


df_policy.groupby("Policy_Number")[['qpt', 'lia_class', 'plia_acc', 'pdmg_acc']].nunique().describe()


# In[33]:


df_policy.groupby("Policy_Number")['Multiple_Products_with_TmNewa_(Yes_or_No?)'].max().value_counts(dropna=False)[:10]


# In[34]:


(df_policy["ibirth"] == df_policy["dbirth"]).sum() / df_policy.shape[0]


# #### Count Unique Values

# In[35]:


df_policy.fillna(-1).nunique()


# In[36]:


df_policy[['Policy_Number', 'fsex']].drop_duplicates()['fsex'].value_counts(dropna=False)


# In[37]:


df_policy[['Policy_Number', 'Imported_or_Domestic_Car']].drop_duplicates()['Imported_or_Domestic_Car'].value_counts(dropna=False)


# #### Investigate Cancellation

# In[38]:


df_cancellation = df_policy[["Policy_Number", "Cancellation"]].drop_duplicates()
print(df_cancellation.shape, df_policy.shape)
df_cancellation = df_cancellation.merge(df_train, on="Policy_Number")
df_cancellation["is_zero"] = df_cancellation["Next_Premium"] == 0
df_cancellation.groupby("Cancellation")[["is_zero"]].agg(["mean", "count"])


# ### Check Claims

# Unique column values:

# In[39]:


df_claims.fillna(-1).nunique()


# In[40]:


# Take total paid in log scale
df_claims['Total_Paid'] = np.log1p(df_claims["Paid_Loss_Amount"] + df_claims["paid_Expenses_Amount"])
fig = plt.figure(figsize=(10, 5))
ax = sns.distplot(
    df_claims['Total_Paid'], bins=100, color="skyblue", label="1", 
    kde=False)
ax.set_title("Total_Paid (in log scale)")


# At_Fault values are noisy:

# In[41]:


df_claims['At_Fault?'].value_counts()


# Make it simpler:

# In[42]:


df_claims.loc[df_claims['At_Fault?'] > 100, 'At_Fault?'] = 100
df_claims.loc[(df_claims['At_Fault?'] > 0) & (df_claims['At_Fault?'] < 100), 'At_Fault?'] = 50
df_claims['At_Fault?'].value_counts()


# In[43]:


tmp = df_claims[['At_Fault?', 'Total_Paid']]
sns.jointplot(x=tmp["At_Fault?"], y=tmp['Total_Paid'], kind="hex", height=10)


# Claim_Status is basically useless:

# In[44]:


df_claims['Claim_Status_(close,_open,_reopen_etc)'].value_counts()


# Did not drill deeper on this feature:

# In[45]:


df_claims['DOB_of_Driver'].head()


# ### Examine Coverage

# In[46]:


np.sort(df_claims['Coverage'].unique())


# In[47]:


np.sort((df_policy['Main_Insurance_Coverage_Group'] + "_" + df_policy['Insurance_Coverage']).unique())


# Take a look at coverage statistics at the policy side:

# In[48]:


cnt = df_policy.groupby(["Main_Insurance_Coverage_Group", "Insurance_Coverage"]).size()
cnt[cnt > 5000]


# Mappings from Insurance_Coverage to Main_Insurance_Coverage_Group:

# In[49]:


mappings = {k:v for _, (v,k) in df_policy[["Main_Insurance_Coverage_Group", "Insurance_Coverage"]].drop_duplicates().iterrows()}
mappings


# "竊盜" has almost no claims:

# In[50]:


df_claims["main_coverage_group"] = df_claims['Coverage'].apply(lambda x: mappings[x])
df_claims["main_coverage_group"].value_counts()


# Now the coverage statistics at the claim side:

# In[51]:


cnt = df_claims.groupby(["main_coverage_group", "Coverage"]).size()
cnt[cnt > 1000]


# ### Vehicle Manufature Date

# In[52]:


np.sort(df_policy['Manafactured_Year_and_Month'].unique())


# ### Check Train / Test Split

# Find the unique policies and keep track of their order in file:

# In[53]:


df_uniq_policy = df_policy[["Policy_Number"]].drop_duplicates()
df_uniq_policy["index"] = range(df_uniq_policy.shape[0])
df_uniq_policy.head()


# Label the train and the test dataset:

# In[54]:


test_indices = df_uniq_policy.merge(df_test[["Policy_Number"]], on="Policy_Number")
train_indices = df_uniq_policy.merge(df_train[["Policy_Number"]], on="Policy_Number")
test_indices["dataset"] = "test"
train_indices["dataset"] = "train"
indices = pd.concat([train_indices, test_indices], axis=0).sort_values("index")
indices.sample(10)


# Now check their distribution in the file:

# In[55]:


fig = plt.figure(figsize=(5,10))
sns.boxplot(x="dataset", y="index", data=indices).set_title('Policy Index distribution')


# In[56]:


fig = plt.figure(figsize=(10, 5))
# "xlim":(indices["index"].min(), indices["index"].max())
sns.distplot(indices[indices.dataset=="train"]["index"], bins=50, color="skyblue", label="train", kde=False, hist_kws={"alpha": 0.3, "range":(indices["index"].min(), indices["index"].max())})
ax = sns.distplot(indices[indices.dataset=="test"]["index"], bins=50,  color="red", label="test", kde=False, hist_kws={"alpha": 0.3, "range":(indices["index"].min(), indices["index"].max())})
ax.legend()
ax.set_title("Policy Index Distribution")


# In[57]:


indices[indices.dataset=="test"]["index"].min()


# ### Check Categoricals in Train/Test
# Need to find values that are exclusive to the test dataset.

# In[58]:


df_policy_split = indices[["Policy_Number", "dataset"]].merge(df_policy, on="Policy_Number")


# In[59]:


for col in [
    'Cancellation', 'Vehicle_Make_and_Model1',
    'Vehicle_Make_and_Model2', 'Manafactured_Year_and_Month',
    'Engine_Displacement_(Cubic_Centimeter)', 'Imported_or_Domestic_Car',
    'Coding_of_Vehicle_Branding_&_Type', 'qpt', 'fpt',
    'Main_Insurance_Coverage_Group', 'Insurance_Coverage',
    'Distribution_Channel',
    'Multiple_Products_with_TmNewa_(Yes_or_No?)', 'lia_class', 'plia_acc',
    'pdmg_acc', 'fassured', 'ibirth', 'fsex', 'fmarriage', 'aassured_zip',
    'iply_area', 'dbirth', 'fequipment1', 'fequipment2', 'fequipment3',
    'fequipment4', 'fequipment5', 'fequipment6', 'fequipment9',
    'nequipment9']:
    print(col, len(
        set(df_policy_split[df_policy_split.dataset == "test"][col].unique()) - 
        set(df_policy_split[df_policy_split.dataset == "train"][col].unique())
    ))
    # print(df_policy_split.groupby(["dataset"])[col].nunique())
    print("=" * 20)


# Take a closer look:

# In[60]:


for col in [
    'Vehicle_Make_and_Model1', 'Vehicle_Make_and_Model2', 'Manafactured_Year_and_Month',
    'Engine_Displacement_(Cubic_Centimeter)', 
    'Coding_of_Vehicle_Branding_&_Type', 'qpt',
    'Insurance_Coverage', 'Distribution_Channel',
    'Multiple_Products_with_TmNewa_(Yes_or_No?)', 
    'ibirth', 'dbirth', 'fequipment5']:
    only_in_test = (set(df_policy_split[df_policy_split.dataset == "test"][col].unique()) - 
        set(df_policy_split[df_policy_split.dataset == "train"][col].unique()))
    print(df_policy_split[df_policy_split[col].isin(only_in_test)].groupby(col).size().sort_values(ascending=False))
    print("=" * 20)


# #### Label Encoder
# Maybe setting the appropirate min_obs can make the problem go away (not entirely, though):

# In[61]:


POLICY_FIXED_CATEGORICALS = [
    'Imported_or_Domestic_Car', 'Vehicle_Make_and_Model1',
    'Distribution_Channel',
    'lia_class', 'plia_acc', 'pdmg_acc',
    'fassured',  'iply_area', 'aassured_zip',
    # 'fequipment1', 'fequipment2', 'fequipment3',
    'Multiple_Products_with_TmNewa_(Yes_or_No?)'
]
encoder = LabelEncoder(min_obs=5000)
df_policy_fixed_categoricals = df_policy[
    ["Policy_Number"] + POLICY_FIXED_CATEGORICALS
].drop_duplicates().set_index("Policy_Number")
df_policy_fixed_categoricals = encoder.fit_transform(
    df_policy_fixed_categoricals[POLICY_FIXED_CATEGORICALS]
)


# In[62]:


df_policy_split = indices[["Policy_Number", "dataset"]].drop_duplicates().set_index("Policy_Number").join(
    df_policy_fixed_categoricals)


# Seems like we did it:

# In[63]:


for col in POLICY_FIXED_CATEGORICALS:
    only_in_test = (set(df_policy_split[df_policy_split.dataset == "test"][col].unique()) - 
        set(df_policy_split[df_policy_split.dataset == "train"][col].unique()))
    print(df_policy_split[df_policy_split[col].isin(only_in_test)].groupby(col).size().sort_values(ascending=False))
    print("=" * 20)


# Now do some double-checkingm:

# In[64]:


for col in POLICY_FIXED_CATEGORICALS:
    print(col, df_policy_split[col].nunique())


# In[65]:


for col in POLICY_FIXED_CATEGORICALS:
    print(df_policy_split.groupby([col, "dataset"]).size())
    print("="*20)


# ### Check Claim Date Distributions

# Parse the date string:

# In[66]:


df_claims["Date"] = df_claims["Accident_Date"].apply(lambda x: datetime.strptime(x, "%Y/%m"))
df_claims["Date"].head()


# In[67]:


df_claims["Date"].min(), df_claims["Date"].max()


# Turn the dates into *int64* numbers:

# In[68]:


df_claim_dates = df_claims.groupby("Policy_Number")["Date"].agg(["max", "min"]).reset_index()
for col in ["max", "min"]:
    df_claim_dates[col] = df_claim_dates[col].astype("int64") / 10**9 / 60 / 60 / 24 # ns / s / m / d
df_claim_dates = df_claim_dates.merge(indices, on="Policy_Number")
df_claim_dates["color"] = "green"
df_claim_dates.loc[df_claim_dates.dataset=="test", "color"] = "red"
df_claim_dates.head()


# Plot the minimum claim/accident date against the policy index.
# 
# Take 5000 samples to make a scatterplot:

# In[69]:


fig = plt.figure(figsize=(18, 8))
sample = df_claim_dates.sample(5000)
ax = sns.regplot(
    x=sample["index"], 
    y=sample["min"].astype("int32"), 
    scatter_kws={'c': sample['color'], 'color': None, 'alpha': 0.3})
plt.axvline(x=0, linestyle="dotted", color="red")
plt.axvline(x=95943, linestyle="dotted", color="red")
plt.axvline(x=132800, linestyle="dotted", color="red")
plt.axvline(x=189940, linestyle="dotted", color="red")
plt.axvline(x=252695, linestyle="dotted", color="red")
plt.axvline(x=294932, linestyle="dotted", color="red")
plt.axvline(x=351267, linestyle="dotted", color="red")


# We can also plot the full dataset:

# In[70]:


sample = df_claim_dates.sample(10000)
sns.jointplot(x=sample["index"], y=sample["max"].astype("int32"), kind="hex", height=10)


# #### Check Insurance Coverage
# (Sorry this section is somewhat out of place)

# Check if a policy has only one type of insurance coverage:

# In[71]:


df_n_main_coverages = df_policy.groupby("Policy_Number")[
    "Main_Insurance_Coverage_Group"
].nunique().to_frame("coverages").reset_index()
df_claim_dates = df_claim_dates.merge(df_n_main_coverages, on="Policy_Number")
df_claim_dates["color"] = "green"
df_claim_dates.loc[df_claim_dates["coverages"]==2, "color"] = "blue"
df_claim_dates.loc[df_claim_dates["coverages"]==3, "color"] = "red"


# Apparantly not:

# In[72]:


tmp = indices.merge(df_n_main_coverages, on="Policy_Number")
tmp.head(20)


# ### Check Other Feature Distributions

# #### Insurance Coverage

# In[73]:


cnt = df_policy.Insurance_Coverage.value_counts()
cnt.head()


# In[74]:


tmp = indices.merge(df_policy[["Policy_Number", "Insurance_Coverage"]].drop_duplicates(), on="Policy_Number")
tmp.head()


# In[75]:


N = 5
fig = plt.figure(figsize=(10, 5))
ax = plt.subplot(111)
_ = ax.hist(
    [
        tmp[tmp.Insurance_Coverage==i]["index"] for i in cnt.index[:N]
    ], 100, 
    label = range(N),
    stacked=True, density=True)
plt.title("Insurance_Coverage")
ax.legend()


# #### Manafactured_Year_and_Month

# In[76]:


cnt = df_policy.Manafactured_Year_and_Month.value_counts()
cnt.head()


# In[77]:


tmp = indices.merge(df_policy[["Policy_Number", "Manafactured_Year_and_Month"]].drop_duplicates(), on="Policy_Number")
tmp.head()


# In[78]:


N = 5
fig = plt.figure(figsize=(10, 5))
ax = plt.subplot(111)
_ = ax.hist(
    [
        tmp[tmp.Manafactured_Year_and_Month==i]["index"] for i in cnt.index[:N]
    ], 100, 
    label = range(N),
    stacked=True, density=True)
plt.title("Manafactured_Year_and_Month")
ax.legend()


# #### Has prior policy

# In[79]:


df_policy["has_prior"] = df_policy.Prior_Policy_Number.isnull()
cnt = df_policy.has_prior.value_counts()
cnt


# In[80]:


tmp = indices.merge(df_policy[["Policy_Number", "has_prior"]].drop_duplicates(), on="Policy_Number")
tmp.head()


# In[81]:


N = 3
fig = plt.figure(figsize=(10, 5))
ax = plt.subplot(111)
_ = ax.hist(
    [
        tmp[tmp.has_prior==i]["index"] for i in cnt.index[:N]
    ], 100, 
    label = range(N),
    stacked=True, density=True)
plt.title("has_prior")
ax.legend()


# #### Distribution Channel

# In[82]:


cnt = df_policy.Distribution_Channel.value_counts()
cnt.head()


# In[83]:


tmp = indices.merge(df_policy[["Policy_Number", "Distribution_Channel"]].drop_duplicates(), on="Policy_Number")
tmp.head()


# In[84]:


N = 10
fig = plt.figure(figsize=(10, 5))
ax = plt.subplot(111)
_ = ax.hist(
    [
        tmp[tmp.Distribution_Channel==i]["index"] for i in cnt.index[:N]
    ], 100, 
    label = range(N),
    stacked=True, density=True)
plt.title("Distribution Channel")
ax.legend()


# In[85]:


tmp[tmp.Distribution_Channel==cnt.index[1]]["index"].max()


# Find the splitting point:

# In[86]:


fig = plt.figure(figsize=(10, 5))
ax = plt.subplot(111)
subset = tmp[tmp.Distribution_Channel==cnt.index[5]]["index"]
subset = subset[(subset > 90000) & (subset < 150000)]
_ = ax.hist(
    subset, 1000, 
    density=True)
plt.axvline(x=95943, linestyle="dotted", color="red")
plt.axvline(x=132800, linestyle="dotted", color="red")


# #### iply_area

# In[87]:


cnt = df_policy.iply_area.value_counts(dropna=False)
cnt.head()


# First only check test policies:

# In[88]:


# tmp = indices.merge(df_policy[["Policy_Number", "iply_area"]].drop_duplicates(), on="Policy_Number")
tmp = indices.merge(df_policy[["Policy_Number", "iply_area"]][
    df_policy.Policy_Number.isin(df_test.Policy_Number)].drop_duplicates(), on="Policy_Number")
tmp.head()


# In[89]:


N = 10
fig = plt.figure(figsize=(10, 5))
ax = plt.subplot(111)
_ = ax.hist(
    [
        tmp[tmp.iply_area==i]["index"] for i in cnt.index[:N]
    ], 100, 
    label = range(N),
    stacked=True, density=True)
plt.title("iply_area")
ax.legend()


# Find the splitting point:

# In[90]:


fig = plt.figure(figsize=(10, 5))
ax = plt.subplot(111)
subset = tmp[tmp.iply_area==cnt.index[3]]["index"]
_ = ax.hist(
    subset, 1000, 
    density=True)
plt.axvline(x=95943, linestyle="dotted", color="red")
plt.axvline(x=132800, linestyle="dotted", color="red")
plt.axvline(x=189940, linestyle="dotted", color="red")
plt.axvline(x=252695, linestyle="dotted", color="red")


# In[91]:


subset[subset < 200000].max()


# In[92]:


subset.max()


# In[93]:


fig = plt.figure(figsize=(10, 5))
ax = plt.subplot(111)
subset = tmp[tmp.iply_area==cnt.index[8]]["index"]
subset = subset[(subset < 295000) & (subset > 250000)]
_ = ax.hist(
    subset, 1000, 
    density=True)
# plt.axvline(x=95943, linestyle="dotted", color="red")
# plt.axvline(x=132800, linestyle="dotted", color="red")
# plt.axvline(x=189940, linestyle="dotted", color="red")
plt.axvline(x=252695, linestyle="dotted", color="red")
plt.axvline(x=294932, linestyle="dotted", color="red")


# In[94]:


subset.max()


# Now check the full dataset:

# In[95]:


tmp = indices.merge(df_policy[["Policy_Number", "iply_area"]][
    df_policy.Policy_Number.isin(df_train.Policy_Number)].drop_duplicates(), on="Policy_Number")
N = 5
fig = plt.figure(figsize=(10, 5))
ax = plt.subplot(111)
_ = ax.hist(
    [
        tmp[tmp.iply_area==i]["index"] for i in cnt.index[:N]
    ], 100, 
    label = range(N),
    stacked=True, density=True)
plt.title("iply_area")
ax.legend()


# #### Distribution_Channel + iply_area

# In[96]:


df_policy["combo"] = df_policy["Distribution_Channel"].astype("str") + "_" + df_policy["iply_area"].astype("str")
cnt = df_policy.combo.value_counts(dropna=False)
cnt.head(10)


# In[97]:


tmp = indices.merge(df_policy[["Policy_Number", "combo"]].drop_duplicates(), on="Policy_Number")
tmp.head()


# In[98]:


N = 10
fig = plt.figure(figsize=(10, 5))
ax = plt.subplot(111)
_ = ax.hist(
    [
        tmp[tmp.combo==i]["index"] for i in cnt.index[:N]
    ], 100, 
    label = range(N),
    stacked=True, density=True)
plt.title("combo")
ax.legend()


# #### fmarriage

# In[99]:


df_policy["fmarriage"].fillna(" ", inplace=True)
cnt = df_policy.fmarriage.value_counts(dropna=False)
cnt


# In[100]:


tmp = indices.merge(df_policy[["Policy_Number", "fmarriage"]].drop_duplicates(), on="Policy_Number")
tmp.head()


# In[101]:


N = 3
fig = plt.figure(figsize=(10, 5))
ax = plt.subplot(111)
_ = ax.hist(
    [
        tmp[tmp.fmarriage==i]["index"] for i in cnt.index[:N]
    ], 100, 
    label = range(N),
    stacked=True, density=True)
plt.title("fmarriage")
ax.legend()


# #### aassured_zip

# In[102]:


cnt = df_policy.aassured_zip.value_counts()
cnt.head(20)


# In[103]:


tmp = indices.merge(df_policy[["Policy_Number", "aassured_zip"]].drop_duplicates(), on="Policy_Number")
tmp.head()


# In[104]:


N = 8
fig = plt.figure(figsize=(10, 5))
ax = plt.subplot(111)
_ = ax.hist(
    [
        tmp[tmp.aassured_zip==i]["index"] for i in cnt.index[:N]
    ], 100, 
    label = range(N),
    stacked=True, density=True)
plt.title("aassured_zip")
ax.legend()


# #### qpt

# In[105]:


df_policy.qpt.value_counts()


# In[106]:


tmp = indices.merge(df_policy[["Policy_Number", "qpt"]].drop_duplicates(), on="Policy_Number")
tmp.head()


# In[107]:


fig = plt.figure(figsize=(10, 5))
ax = plt.subplot(111)
_ = ax.hist(
    [
        tmp[tmp.qpt==5]["index"],
        tmp[tmp.qpt==7]["index"],
        tmp[tmp.qpt==3]["index"]
    ], 100, 
    label = [5, 7, 3],
    stacked=True, density=True)
plt.title("qpt")
ax.legend()


# In[108]:


df_policy.Cancellation.value_counts()


# #### Cancellation

# In[109]:


tmp = indices.merge(df_policy[["Policy_Number", "Cancellation"]].drop_duplicates(), on="Policy_Number")
tmp.head()


# In[110]:


fig = plt.figure(figsize=(10, 5))
ax = plt.subplot(111)
_ = ax.hist(
    [
        tmp[tmp.Cancellation==" "]["index"],
        tmp[tmp.Cancellation=="Y"]["index"]
    ], 100, 
    label = ['N', 'Y'],
    stacked=True, density=True)
plt.title("Cancellation")
ax.legend()


# In[111]:


df_policy.iply_area.value_counts()


# #### Imported_or_Domestic_Car

# In[112]:


df_policy.Imported_or_Domestic_Car.value_counts(dropna=False)


# In[113]:


tmp = indices.merge(df_policy[["Policy_Number", "Imported_or_Domestic_Car"]].drop_duplicates(), on="Policy_Number")
tmp.head()


# In[114]:


fig = plt.figure(figsize=(10, 5))
plt.style.use('seaborn')
_ = plt.hist(
    [
        tmp[tmp.Imported_or_Domestic_Car==10]["index"],
        tmp[tmp.Imported_or_Domestic_Car==30]["index"],
        tmp[tmp.Imported_or_Domestic_Car==40]["index"],
        tmp[~tmp.Imported_or_Domestic_Car.isin((10, 30, 40))]["index"],
    ], 100, 
    label = [10, 30, 40, "other"],
    stacked=True, density=True)
ax = plt.subplot(111)
ax.legend()
plt.title("Imported_or_Domestic_Car")


# #### lia_class

# In[115]:


df_policy.lia_class.value_counts(dropna=False)


# In[116]:


tmp = indices.merge(df_policy[["Policy_Number", "lia_class"]].drop_duplicates(), on="Policy_Number")
tmp.head()


# In[117]:


fig = plt.figure(figsize=(10, 5))
plt.style.use('seaborn')
_ = plt.hist(
    [
        tmp[tmp.lia_class==-1]["index"],
        tmp[tmp.lia_class==1]["index"],
        tmp[tmp.lia_class==2]["index"],
        tmp[tmp.lia_class==0]["index"],
        tmp[tmp.lia_class>2]["index"]
    ], 
    100, label=[-1, 1, 2, 0, "other"], 
    stacked=True, density=True)
ax = plt.subplot(111)
ax.legend()
plt.title("lia_class")


# #### fsex

# In[118]:


df_policy["fsex"].fillna(" ",inplace=True)
df_policy["fsex"].unique()


# In[119]:


tmp = indices.merge(df_policy[["Policy_Number", "fsex"]].drop_duplicates(), on="Policy_Number")
tmp.head()


# In[120]:


fig = plt.figure(figsize=(10, 5))
plt.style.use('seaborn')
_ = plt.hist(
    [
        tmp[tmp.fsex=='1']["index"],
        tmp[tmp.fsex=='2']["index"],
        tmp[tmp.fsex==' ']["index"],
    ], 100, label=["1", "2", "others"],
    stacked=True, density=True)
ax = plt.subplot(111)
ax.legend()
plt.title("fsex")


# #### fassured

# In[121]:


df_policy["fassured"].unique()


# In[122]:


tmp = indices.merge(df_policy[["Policy_Number", "fassured"]].drop_duplicates(), on="Policy_Number")
tmp.head()


# In[123]:


df_claim_dates = df_claim_dates.merge(df_policy[["Policy_Number", "fassured"]].drop_duplicates(), on="Policy_Number")


# In[124]:


df_claim_dates["color"] = "green"
df_claim_dates.loc[df_claim_dates["fassured"]==2, "color"] = "blue"
df_claim_dates.loc[df_claim_dates["fassured"]==3, "color"] = "red"
df_claim_dates.loc[df_claim_dates["fassured"]==6, "color"] = "purple"


# In[125]:


tmp.fassured.value_counts()


# In[126]:


fig = plt.figure(figsize=(10, 5))
plt.style.use('seaborn')
_ = plt.hist(
    [
        tmp[tmp.fassured==1]["index"],
        tmp[tmp.fassured==2]["index"],
        tmp[tmp.fassured==3]["index"],
        tmp[tmp.fassured==6]["index"],
    ],
    100, stacked=True, density=True)
ax = plt.subplot(111)
ax.legend()
plt.title("fassured")

