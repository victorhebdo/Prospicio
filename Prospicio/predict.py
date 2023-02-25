import pandas as pd
from prospicio.registry import load_model
from prospicio.model import MultiLabelBinarizer, pipeline

#model = load_model()

def predict(X_new):
    # maybe need to add some prep work to get X in the right
    # format (format same as X that went into the pipeline,
    # i.e. after preprocessing work)

    mlb_2 = MultiLabelBinarizer(sparse_output=True)

    X_new_ind_cleaned = X_new.join(
            pd.DataFrame.sparse.from_spmatrix(
                mlb_2.fit_transform(X_new['industries_cleaned']),
                index=X_new.index,
                columns=mlb_2.classes_))

    pipeline.transform(X_new_ind_cleaned)
    model = load_model
    my_prediction = model.predict(X_new_ind_cleaned)
    print(my_prediction)

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
    print(X_new)
    #columns=['country_code', 'employee_range', 'min_revenues', 'traffic.monthly', 'industries_cleaned'])
