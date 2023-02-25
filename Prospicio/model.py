import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

# sklearn preproc
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder

from sklearn.model_selection import cross_val_score,  cross_validate

# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.linear_model import Ridge, Lasso, LinearRegression
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.svm import SVR
# from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# from sklearn.ensemble import AdaBoostRegressor
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import VotingRegressor
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.ensemble import StackingRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from prospicio.data import get_data
from prospicio.registry import save_model
from prospicio.multicatencoder import MultiCategoriesEncoder


def preproc_pipeline():
    columns_to_num = ['traffic.monthly', 'min_revenues']
    columns_to_ohe = ['employee_range', 'country_code']
    column_to_vect = ['industries_cleaned']

    preproc_numerical_baseline = make_pipeline(
        SimpleImputer(),
        StandardScaler())

    preproc_categorical_baseline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore"))

    preproc_multicat_baseline = make_pipeline(
        MultiCategoriesEncoder()
    )

    preproc_baseline = make_column_transformer(
        (preproc_numerical_baseline, columns_to_num),
        (preproc_categorical_baseline, columns_to_ohe),
        (preproc_multicat_baseline, column_to_vect),
        remainder="drop")
    return preproc_baseline


def pipeline():
    pipe_baseline = make_pipeline(preproc_pipeline(), LogisticRegression())
    return pipe_baseline


def train(X, y):
    model = pipeline()
    model.fit(X, y)
    save_model(model)


if __name__ == "__main__":
    X, y = get_data()
    score_baseline = cross_val_score(pipeline(), X, y, cv=5).mean()
    print(score_baseline)
    train(X, y)
    """
    test_preproc = preproc_pipeline()
    X_transformed = pd.DataFrame(test_preproc.fit_transform(X),
                                 columns=test_preproc.get_feature_names_out())
    print(X_transformed.columns)
    """
