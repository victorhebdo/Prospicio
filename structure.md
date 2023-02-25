# Back-end repo = Prospicio

## Folder prospicio
### data.py
```python
def read_file():
  ...
  return df

def preprocessing(crunch):
  ...
  return X, y

def get_data():
  data = read_file()
  return preprocessing(data)
```

### model.py
```python
from prospicio.registry import save_model

def make_pipeline():
  preproc_baseline = ...
  model = LogisticRegression()
  pipe_baseline = make_pipeline(preproc_baseline, model)
  return pipe_baseline

def train(X, y):
  pipeline = make_pipeline()
  pipeline.fit(X, y)
  save_model(pipeline)
```

### predict.py
```python
from prospicio.registry import load_model

model = load_model('models/model.joblib')

def predict(X):
  # maybe need to add some prep work to get X in the right
  # format (format same as X that went into the pipeline,
  # i.e. after preprocessing work)
  return model.predict(X)
```

### registry.py
```python
import joblib
def save_model(model):
  model_path = os.path.join('models/model.joblib')
  joblib.dump(model, model_path)

def load_model(model_path):
  return joblib.load(model_path)
```




## Folder api
### fast.py
```python
...
from prospicio.predict import predict
...
@app.get("/predict")
def get_predict(input_one,
                input_two,
                input_three):
    # TODO:
    # convert to right formats
    # make a dataframe from the inputs in the format for the predict
    df = pd.DataFrame(...)
    prediction = predict(df)
    return {
        'prediction': prediction
    }
```

# Front-end repo = Prospicio-Front (?)

## app.py
```python
import streamlit
import requests

# TODO some fancy stuff with streamlit to get the inputs

base_url = "https://localhost:8000"
url = base_url + "/predict"
params = {
  'input_one': ...,
  'input_two': ...
}

prediction = requests.get(url=url, params=params).json()['prediction']

# TODO show the prediction
