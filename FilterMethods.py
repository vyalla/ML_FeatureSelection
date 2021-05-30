# %% [markdown]
# ### Import necessary packages
# 

# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %%
df = pd.read_csv('data/housing.csv')
print('Shape: ', df.shape)
print(df.head())

# %% [markdown]
# ### Display the data types of the dataframe

# %%
print(df.dtypes)

# %% [markdown]
# ### Split the data in to train, test split. In this housing data, "SalePrice" is the dependent/target feature.

# %%
from sklearn.model_selection import train_test_split
X_train, y_train, X_test, y_test = train_test_split(df.drop(['SalePrice'], axis=1), df['SalePrice'], test_size=0.3, random_state=0)

# %% [markdown]
# ### Select the numeric columns only

# %%
numeric_X_train = X_train[X_train.select_dtypes([np.number]).columns]

print(len(numeric_X_train.columns))
print(numeric_X_train.columns)
print(numeric_X_train.head())

# %% [markdown]
# ### Use VarianceThreshold feature selector to select the feature which have more
# ### variance i.e more than zero

# %%
from sklearn.feature_selection import VarianceThreshold

vs_constants = VarianceThreshold(threshold=0)
vs_constants.fit(numeric_X_train)

print(len(vs_constants.get_support()))
print(vs_constants.get_support())

# %% [markdown]
# ### Get all the selected column names

# %%
constant_columns = [column for column in numeric_X_train 
                    if column not in numeric_X_train.columns[vs_constants.get_support()]]
#
print('Lenght of X train columns: ', len(X_train.columns))
print('Lenght of numeric X train columns: ', len(numeric_X_train.columns))
print('Lenght of constant columns: ', len(constant_columns))

# %%
