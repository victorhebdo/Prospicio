import pandas as pd
from prospicio.registry import load_model

model = load_model()


def predict(X_new):
    """
    Requires pd.Dataframe of 5 columns =
    tmp = [{
        "country_code": "de",
        "employee_range": 75,
        "min_revenues": 8.084688e+06,
        "traffic.monthly":45023.0,
        "industries_cleaned":{'Software', 'Business_Products_And_Services'}
    }]
    """
    my_prediction = model.predict(X_new)
    return my_prediction


if __name__ == "__main__":
    tmp = [{
        "country_code": "de",
        "employee_range": 75,
        "min_revenues": 8.084688e+06,
        "traffic.monthly":45023.0,
        "industries_cleaned":{'Software', 'Business_Products_And_Services'}
    }]
    X_new = pd.DataFrame(data=tmp)
    print(predict(X_new))
