from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer


class MultiCategoriesEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.label_transformer = CountVectorizer(analyzer=set)

    def fit(self, X, y=None):
        self.label_transformer.fit(X.squeeze())
        return self

    def transform(self, X, y=None):
        return self.label_transformer.transform(X.squeeze(axis=1)).toarray()

    def get_feature_names_out(self, X):
        return self.label_transformer.get_feature_names_out()
