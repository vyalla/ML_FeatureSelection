# %% [markdown]
# ## Import necessary packages and libraries

# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier

# %%
# Load the Housing data
df = pd.read_csv('./data/housing.csv')
print(df.head())

# %%
# Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(df.drop(labels=['SalePrice'], axis=1), df['SalePrice']
                                    , random_state=0, test_size=0.3)
print(X_train.head())
print(y_train.head())

# %% [markdown]
# ## Forward Feature Selection

# %%
sfs = SequentialFeatureSelector(RandomForestClassifier(), k_features=10, forward=True, 
        floating=False, scoring='accuracy', cv=2)
# fit the object to the training data
sfs = sfs.fit(X_train, y_train)

# %%
# Print the selected features.
selected_features = X_train.columns[list(sfs.k_feature_idx_)]
print(selected_features)

# %%
# Print final prediction score
print(sfs.k_score_)

# %%
# Transform the newly selected features
X_train_sfs = sfs.transform(X_train)
X_test_sfs = sfs.transform(X_test)

# %%
print(X_test_sfs)
print(X_test_sfs[0, :])
print(X_test_sfs[0])

# %% [markdown]
# ## Backward Feature Selection

# %%
sbs = SequentialFeatureSelector(RandomForestClassifier(), k_features=10, forward=False
        , floating=False, scoring="accuracy", cv=2)
sbs = sbs.fit(X_train, y_train)

# %%
selected_features = X_train.columns[list(sbs.k_feature_idx_)]
print(selected_features)
print(sbs.k_score_)

# %%
X_train_sbs = sbs.transform(X_train)
X_test_sbs = sbs.transform(X_test)

# %%
print(X_test_sbs)
# %%
