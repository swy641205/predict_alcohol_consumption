import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

class KNNRegression:
    def __init__(self, df: pd.DataFrame, target: str, n_neighbors=5):
        self.df = df
        self.target = target
        self.n_neighbors = n_neighbors
        self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        self.numerical_cols = df.select_dtypes(exclude=['object']).columns.tolist()
        self.numerical_cols.remove(self.target)
        self.model = self.train_model()

    def train_model(self):
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]
        
        numerical_transformer = StandardScaler()
        
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_cols),
                ('cat', categorical_transformer, self.categorical_cols)
            ])
        
        knn_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', KNeighborsRegressor(n_neighbors=self.n_neighbors))
        ])
        
        knn_pipeline.fit(X, y)
        
        return knn_pipeline

    def predict(self, inputs):
        input_df = pd.DataFrame([inputs])
        
        current_categorical_cols = [col for col in self.categorical_cols if col in inputs]
        current_numerical_cols = [col for col in self.numerical_cols if col in inputs]
        
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, current_numerical_cols),
                ('cat', categorical_transformer, current_categorical_cols)
            ])
        
        knn_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', KNeighborsRegressor(n_neighbors=self.n_neighbors))
        ])
        
        knn_pipeline.fit(self.df[current_numerical_cols + current_categorical_cols], self.df[self.target])
        
        input_df = input_df[current_numerical_cols + current_categorical_cols]
        prediction = knn_pipeline.predict(input_df)
        
        return prediction[0]

    def evaluate_model(self, cv=5):
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]
        
        numerical_transformer = StandardScaler()
        
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_cols),
                ('cat', categorical_transformer, self.categorical_cols)
            ])
        
        knn_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', KNeighborsRegressor(n_neighbors=self.n_neighbors))
        ])
        
        scores = cross_val_score(knn_pipeline, X, y, cv=cv, scoring='neg_mean_squared_error')
        rmse_scores = (-scores) ** 0.5
        mean_rmse = rmse_scores.mean()
        std_rmse = rmse_scores.std()
        return mean_rmse, std_rmse
