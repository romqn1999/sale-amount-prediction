import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class LagFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, lags, col):
        self.lags = lags
        self.col = col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for lag in self.lags:
            shifted = X[['date_block_num', 'shop_id', 'item_id', self.col]].copy()
            shifted['date_block_num'] += lag
            shifted = shifted.rename(columns={self.col: f'{self.col}_lag_{lag}'})
            X = pd.merge(X, shifted, on=['date_block_num', 'shop_id', 'item_id'], how='left')
        return X

class FillNA(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.fillna(0)
