import pandas as pd
import numpy as np
import json
import os

from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.linear_model import LinearRegression

from prospicio.industry_dict import total_sectors_n_industries

PACKAGE_DIR = os.path.dirname(os.path.dirname(__file__))
RAW_DATA_DIR = os.path.join(PACKAGE_DIR, "raw_data")

# reading the CSV file
def read_file(): # TODO change path
    my_path = os.path.join(RAW_DATA_DIR, 'crunchbase.csv')
    csvFile = pd.read_csv(my_path)
    return csvFile


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
    # TODO Change this to avoid retraining on my new data
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


def get_data():
    df = read_file()
    df = preprocessing(df)
    # Undersampling
    funds_binary_1 = df[df['funds_binary']==1]
    funds_binary_2 = df[df['funds_binary']==0]
    funds_binary_2 = funds_binary_2.sample(round(len(funds_binary_1)*1.5))
    reduced_df = pd.concat([funds_binary_1, funds_binary_2], axis=0).sample(frac=1)
    # Splitting X, y attention, industries_list still to multilabel
    X = reduced_df.drop(columns=['funds_binary','_id', 'series.total', 'industries_list'])
    y = reduced_df['funds_binary']
    return X, y


if __name__ == "__main__":
    X, y = get_data()
    print(X.shape)
    print(X.columns)
    print(X.head(1))
