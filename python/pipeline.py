# %%
import pandas as pd
import numpy as np
import Orange

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

from Orange.preprocess import Discretize, Impute, Continuize
from Orange.preprocess.discretize import Binning

def natural_binning(table, n_bins=3):
    # Discretize the continous features
    discretizer = Discretize(method=Binning(n=n_bins))
    discretized_table = discretizer(table)
    return discretized_table

def impute(table):
    # Impute missing values
    imputer = Impute()
    imputed_table = imputer(table)
    return imputed_table

def binarize(table):
    # Continuize the discrete features
    continuizer = Continuize()
    continuized_table = continuizer(table)
    return continuized_table

def table_to_dataframe(table):
    # Convert Orange table to pandas dataframe
    column_names = [attribute.name for attribute in table.domain.attributes]
    df = pd.DataFrame(table.X, columns=column_names)
    return df

def remove_first_column(data):
    # Remove the first column of the dataframe (one of the target variable columns)
    df = data.iloc[:, 1:]
    return df

def feature_selection(data, n_features):
    X, y = data[data.columns[1:]], data[data.columns[0]]
    # Select the best features
    selector = SelectFromModel(estimator=LogisticRegression(penalty='l1', solver="liblinear"), max_features=min(n_features, X.shape[1]))
    X = selector.fit_transform(X, y)
    selected_features = selector.get_support(indices=True)
    selected_features += 1
    selected_features = np.insert(selected_features, 0, 0)
    return data.iloc[:, selected_features]

def my_pipeline(dataframe, n_features=15):
    table = Orange.data.table_from_frame(dataframe)
    table = natural_binning(table)
    table = impute(table)
    table = binarize(table)
    dataframe = table_to_dataframe(table)
    dataframe = remove_first_column(dataframe)
    dataframe = feature_selection(dataframe, n_features)
    return dataframe