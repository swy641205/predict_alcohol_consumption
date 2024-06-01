import statsmodels.formula.api as smf
import pandas as pd


class LinearRegression:
    def __init__(self, df: pd.DataFrame, target: str):
        self.df = df
        self.target = target
        self.coefficients, self.intercept, self.model = self.get_significant()

    def get_significant(self):
        predictors = [i for i in self.df.columns if i != self.target]
        formula_str = f"{self.target} ~ " + " + ".join(predictors)

        model = smf.ols(formula=formula_str, data=self.df).fit()

        intercept = model.params.Intercept
        p_values = model.pvalues
        coef = model.params
        coef_list = {}
        for i in range(len(coef)):
            if p_values.iloc[i] < 0.05:
                coef_list[coef.index[i]] = coef.iloc[i]
        return coef_list, intercept, model

    def predict(self, inputs):
        r = self.intercept
        for key, value in self.coefficients.items():
            r += value * inputs[key]
        return r
    






