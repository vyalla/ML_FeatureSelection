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
# ## Constant features
# ### Split the data in to train, test split. In this housing data, "SalePrice" is the dependent/target feature.

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop(['SalePrice'], axis=1), df['SalePrice'], test_size=0.3, random_state=0)

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
'''
for column in X_train.columns:
    print(X_train[column].dtype)
#
print(X_train['LandContour'].unique())
print(len(X_train['LandContour'].unique()))
'''
constant_categorical_columns = [column for column in X_train.columns
                                if (X_train[column].dtype == "object" and  len(X_train[column].unique()) == 1)]
print("Constant categorical columns:")
print(constant_categorical_columns)
#
all_constant_columns = constant_columns + constant_categorical_columns
print("All Constant columns:")
print(all_constant_columns)

# %% [markdown]
# ### Drop all constant columns from X_train and X_text

# %%
X_train.drop(labels=all_constant_columns, axis=1, inplace=True)
X_test.drop(labels=all_constant_columns, axis=1, inplace=True)


# %% [markdown]
# ## Quasi Constant Features

# %%
threshold = 0.98
quasi_constant_features = []
#
for feature in X_train.columns:   
    value_counts = X_train[feature].value_counts()
    print('Value counts for', feature)
    print(value_counts)
    #Calculate the ratio
    value_percentage = (value_counts / np.float64(len(X_train))).sort_values(ascending=False)
    print('Value percentage:')
    print(value_percentage)
    predominant = value_percentage.values[0]
    #
    #
    if(predominant >= threshold):
        quasi_constant_features.append(feature)
    
    #
print('Quasi constant features:')
print(quasi_constant_features)

# %% [markdown]
# ### Drop the Quasi constant features for X_train and X_test

# %%
X_train.drop(labels=quasi_constant_features, axis=1, inplace=True)
X_test.drop(labels=quasi_constant_features, axis=1, inplace=True)

# %% [markdown]
# ## Duplicated features

# %%
train_features_T = X_train.T
#
print(train_features_T)
# print the number of duplicated features
print(train_features_T.duplicated())
# select the duplicated columns
duplicated_columns = train_features_T[train_features_T.duplicated()].index.values
#
#
X_train.drop(labels=duplicated_columns, axis=1, inplace=True)
X_test.drop(labels=duplicated_columns, axis=1, inplace=True)

# %% [markdown]
# ## Correlation Methods

# %%
correlated_features = set()
correlation_matrix = X_train.corr()
#
plt.figure(figsize=(11, 11))
sns.heatmap(correlation_matrix)

# %%
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            column = correlation_matrix.columns[i]
            correlated_features.add(column)

print(correlated_features)

# %% [markdown]
# ### Drop highly correlated columns

# %%
X_train.drop(labels=correlated_features, axis=1, inplace=True)
X_test.drop(labels=correlated_features, axis=1, inplace=True)
# %%
