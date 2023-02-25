import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import json

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


# sklearn preproc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import make_scorer

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression

from prospicio.industry_dict import total_sectors_n_industries
#from Prospicio.predict import predict
#from sklearn import set_config; set_config(display='diagram')

# reading the CSV file
def read_file():
    csvFile = pd.read_csv('../raw_data/crunchbase.csv')
    return csvFile

# displaying the contents of the CSV file

def show_info(csvFile):
    for (columnName, columnData) in csvFile.items():
        print()
        print('Column Name : ', columnName)
        print('Column Contents : ', columnData.values[:1])
        print('Content types : ', type(columnData.values[:1]))
        print()

def preprocessing(crunch):
    # Remove empty data from csvFile that was not scraped
    crunch = crunch[crunch['scraped']==True]

    # Remove all columns that are not needed for regressiion
    crunch = crunch[['_id', 'country_code', 'employee_range',
                 'industries', 'min_revenues', 'series.total', 'traffic.monthly']]

    # Remove all data where industries is nan
    crunch = crunch[crunch['industries'].notna()]

    # Lambda function to extract json format inside industries column
    extract_dict = lambda row: [lines['name'] for lines in json.loads(row)]
    # Extract all dataframes from column and convert it in a list
    crunch['industries_list']= crunch['industries'].apply(extract_dict)

    # Drop columns again
    crunch.drop(columns = ['industries'], inplace=True)

    # Filter csv to have only valid values on employee_range
    crunch = crunch[crunch['employee_range'].notna()]

    # Lambda function to clean employee range
    employee_transf = lambda x: int(np.mean([ int(a) for a in x[1:-1].split(',')]))
    crunch['employee_range']= crunch['employee_range'].apply(employee_transf)


    # clean industries

    cleaning_ind_complete = lambda row: set( [k for k, v in total_sectors_n_industries.items() if set(row).intersection(set(v)) ])
    crunch['industries_cleaned'] = crunch['industries_list'].apply(cleaning_ind_complete)

    # Change total funds into one to create our target for logistic regressor model 0
    crunch['funds_binary'] = crunch['series.total'].apply(lambda x: int(1) if x>0 else int(0))

    # Linear Regression to predict min_revenues per employees
    crunch_rev = crunch[crunch['min_revenues'].notna()]
    crunch_not_rev = crunch[crunch['min_revenues'].isna()]
    X_reg = crunch_rev[['employee_range']]
    y_reg = crunch_rev['min_revenues']
    reg = LinearRegression().fit(X_reg, y_reg)

    employee_dataframe = crunch_not_rev[['employee_range']]

    test_revenues_from_reg = pd.DataFrame( reg.predict(employee_dataframe), index=employee_dataframe.index, \
                                      columns=['revenues_from_reg'] )
    crunch_not_rev['revenues_from_reg'] = test_revenues_from_reg
    #  Remove columns min_revenues after being transformed
    crunch_not_rev.drop(columns=['min_revenues'], inplace=True)

    # change name
    crunch_not_rev.rename(columns={'revenues_from_reg': 'min_revenues'}, inplace = True)

    # Dataframe with both extracts
    df = pd.concat([crunch_rev, crunch_not_rev], axis=0).sample(frac=1)

    return df

def multi_hot_encoder(df):
    mlb = MultiLabelBinarizer(sparse_output=True)

    df_ind_cleaned = df.join(
            pd.DataFrame.sparse.from_spmatrix(
                mlb.fit_transform(df['industries_cleaned']),
                index=df.index,
                columns=mlb.classes_))

    funds_binary_1_again = df_ind_cleaned[df_ind_cleaned['funds_binary']==1]
    funds_binary_2_again = df_ind_cleaned[df_ind_cleaned['funds_binary']==0]
    funds_binary_2_again = funds_binary_2_again.sample(6000)
    reduced_df = pd.concat([funds_binary_1_again, funds_binary_2_again], axis=0).sample(frac=1)
    X = reduced_df.drop(columns=['funds_binary'])
    X= X.drop(columns=['_id', 'industries_list', 'series.total'])
    y = reduced_df['funds_binary']
    return X, y

def pipeline(X,y):
    # preparing pipeline
    column_to_impute = ['traffic.monthly']
    columns_to_num = ['traffic.monthly', 'min_revenues']
    columns_to_ohe = ['employee_range', 'country_code']

    preproc_numerical_imputer_traffic = make_pipeline(
        SimpleImputer())

    preproc_numerical_baseline = make_pipeline(
        SimpleImputer(),
        StandardScaler())


    preproc_categorical_baseline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore"))

    preproc_baseline = make_column_transformer(
        (preproc_numerical_imputer_traffic, column_to_impute),
        (preproc_numerical_baseline, columns_to_num),
        (preproc_categorical_baseline, columns_to_ohe),
        remainder="drop")
    """
    pipe_baseline = make_pipeline(preproc_baseline, LogisticRegression())
    score_baseline = cross_val_score(pipe_baseline, X, y, cv=5).mean()
    print(score_baseline)

    """
    pipe_baseline = make_pipeline(preproc_baseline)
    pipe_baseline.fit(X)
    pipe_baseline.transform(X)

    print("I AM HERE")
    print()
    print()
    print()
    model_reg = LogisticRegression()
    model_reg.fit(X,y)
    breakpoint()
    print("WHAT IS HAPPENING")
    print("WHAT IS HAPPENING")
    print("WHAT IS HAPPENING")
    print("WHAT IS HAPPENING")

    # TEST WITH NEW VARIABLE d
    d = {
        "country_code": "de",
        "employee_range": 75,
        "min_revenues": 8.084688e+06,
        "traffic.monthly":45023.0,
        "industries_cleaned":{'Software', 'Business_Products_And_Services'}
    }


    X_new = pd.DataFrame(data=d, index=['country_code', 'employee_range', 'min_revenues', 'traffic.monthly',
                                     'industries_cleaned'])
    mlb_2 = MultiLabelBinarizer(sparse_output=True)

    X_new_ind_cleaned = X_new.join(
            pd.DataFrame.sparse.from_spmatrix(
                mlb_2.fit_transform(X_new['industries_cleaned']),
                index=X_new.index,
                columns=mlb_2.classes_))

    pipe_baseline.transform(X_new_ind_cleaned)

    my_prediction = model_reg.predict(X_new_ind_cleaned)
    print(my_prediction)


if __name__ == "__main__":
    csvFile = read_file() #WE BEGIN HERE
    #show_info(csvFile)
    df = preprocessing(csvFile)
    X, y = multi_hot_encoder(df)
    pipeline(X, y)

"""    df_ind_cleaned = multi_hot_encoder(df)
    X, y = take_a_sample(df_ind_cleaned)
    pipeline(X, y)
"""
