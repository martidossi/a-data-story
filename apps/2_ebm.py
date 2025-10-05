import marimo

__generated_with = "0.15.1"
app = marimo.App(width="columns")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Explainable Boosting Machine (EBM)""")
    return


@app.cell
def _():
    import marimo as mo

    import pandas as pd
    import numpy as np

    import seaborn as sns
    import matplotlib.pyplot as plt

    from interpret import show
    from interpret.data import Marginal

    from sklearn.datasets import fetch_california_housing
    from sklearn.datasets import load_diabetes
    return fetch_california_housing, mo, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Load data

    🔗 `fetch_california_housing`: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html
    """
    )
    return


@app.cell
def _(fetch_california_housing, pd):
    df_california = fetch_california_housing(as_frame=True)

    X = pd.DataFrame(df_california.data, columns=df_california.feature_names)
    y = df_california.target
    n = len(y)

    df = X.join(y)
    df
    return X, df_california, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Data exploration""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## California housing
    - `Samples total`: 20640
    - `Dimensionality`: 8
    - `Features`: real
    - `Target`: real
    - `No missing values`

    ---

    The data contains information from the 1990 California census. It does provide an **accessible introductory dataset** for teaching people about the basics of machine learning. 
    _Note_: there are no explicit cleaning steps because this dataset arrives fully cleaned: it's all numerical, has no missing values, and requires no preprocessing in that sense.

    - The target variable is the median house value for California districts, in hundreds of thousands of dollars.


    🔗 **Useful links**:

    - https://www.kaggle.com/datasets/camnugent/california-housing-prices (slightly different dataset, the original one?)
    - https://inria.github.io/scikit-learn-mooc/python_scripts/datasets_california_housing.html (Scikit-learn MOOC)
    - https://medium.com/data-science/metrics-for-uncertainty-evaluation-in-regression-problems-210821761aa (very similar study on same data)
    - https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html
    """
    )
    return


@app.cell
def _(df_california):
    df_california.frame.info()
    return


@app.cell
def _(df_california):
    print(df_california.DESCR)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Model fit""")
    return


@app.cell
def _(ExplainableBoostingRegressor, X, X_calib, train_test_split, y):
    seed = 42
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=seed)

    ebm_model = ExplainableBoostingRegressor(random_state=seed)
    ebm_model.fit(X_train, y_train)
    y_pred_calib = ebm_model.predict(X_calib)
    y_pred_test = ebm_model.predict(X_test)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
