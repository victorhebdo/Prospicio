import pandas as pd
import numpy as np
import math


# reading the CSV file
def read_file():
    csvFile = pd.read_csv('raw_data/crunchbase.csv')
    return csvFile

# displaying the contents of the CSV file

def show_info(csvFile):
    for (columnName, columnData) in csvFile.items():
        print()
        print('Column Name : ', columnName)
        print('Column Contents : ', columnData.values[:1])
        print('Content types : ', type(columnData.values[:1]))
        print()

def treat_column(cell):
    #if not math.isnan(cell):
    #if not np.isnan(cell):
    if cell == []:  
        return cell
    elif cell == np.nan:
        return cell
    else:
        #print(cell)
        breakpoint()
        return pd.read_json(cell)
    # df = pd.read_json(csvFile[cell])
    # return df
#my_list = json.loads(csvFile['acquisitions'][0])
#df = pd.read_json(csvFile['acquisitions'][0]) transformer chaque element pour le sortir de string dict
#df = csvFile[['acquisitions']]
#df.applymap(type).eq(str).all()
# acquisitions    False
# dtype: bool
# ipdb> df.applymap(type).eq(str)
#df.applymap(type).eq(str).value_counts()

#df = csvFile['acquisitions']
#df = csvFile['acquisitions']
#df_bool = df.apply(type).eq(str)


#reso = [i for i, val in enumerate(df_bool) if not val]
#df[reso].isna().sum()
#math.isnan(df.at[8860])

my_list = ['acquisitions', 'employees', 'industries', 'investors', 'techs']

if __name__ == "__main__":
    csvFile = read_file() #WE BEGIN HERE
    show_info(csvFile)
    breakpoint()
    #df = treat_column(csvFile, 'acquisitions')
    #csvFile['acquisitions'].apply(lambda acquisition: if not math.isnan(acquisition pd.read_json(acquisition))
    #df = pd.read_json(csvFile['acquisitions'])
    #csvFile['acquisitions'].apply(treat_column)
