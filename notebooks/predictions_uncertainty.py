import marimo

__generated_with = "0.13.11"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    import sys
    import os

    # Data
    from sklearn.datasets import fetch_california_housing

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
    from sklearn.model_selection import GridSearchCV

    from interpret import show
    from interpret.data import Marginal
    from interpret.glassbox import ExplainableBoostingRegressor

    from sklearn.ensemble import RandomForestRegressor
    from quantile_forest import RandomForestQuantileRegressor

    import lightgbm as rf
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    import shap
    return (
        ExplainableBoostingRegressor,
        GridSearchCV,
        LGBMRegressor,
        Marginal,
        RandomForestQuantileRegressor,
        RandomForestRegressor,
        XGBRegressor,
        fetch_california_housing,
        mean_absolute_percentage_error,
        mean_squared_error,
        norm,
        np,
        pd,
        plt,
        shap,
        show,
        sns,
        train_test_split,
    )


@app.cell
def _():
    #sys.path.append(os.path.abspath("src/"))
    #from dataviz import plot_with_error_bars, true_vs_pred
    #from utils import (
    #    calculate_coverage, model_eval_metrics, get_confidence_interval, get_conformalized_interval
    #)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Data import""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## California housing dataset
    The data contains information from the 1990 California census. It does provide an **accessible introductory dataset** for teaching people about the basics of machine learning.
    _Note_: there are no explicit cleaning steps because this dataset arrives fully cleaned: it's all numerical, has no missing values, and requires no preprocessing in that sense.

    - The target variable is the median house value for California districts, in hundreds of thousands of dollars.


    🔗 **Useful links**:

    - https://www.kaggle.com/datasets/camnugent/california-housing-prices (slightly different dataset, the original one?)
    - https://inria.github.io/scikit-learn-mooc/python_scripts/datasets_california_housing.html (Scikit-learn MOOC)
    - https://medium.com/data-science/metrics-for-uncertainty-evaluation-in-regression-problems-210821761aa (very similar study on same data)
    """
    )
    return


@app.cell
def _(fetch_california_housing, pd):
    # Load the California housing dataset
    california = fetch_california_housing(as_frame=True) # Pandas Objects
    X = pd.DataFrame(california.data, columns=california.feature_names)
    y = california.target
    return X, california, y


@app.cell
def _(X, y):
    X, y
    return


@app.cell
def _(california):
    california.frame.info()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We can see that:

    - the dataset contains 20,640 samples and 8 features;
    - all features are numerical features encoded as floating number;
    - there is no missing values.
    """
    )
    return


@app.cell
def _(Marginal, X, show, y):
    # InterpretML
    exploring_data = Marginal().explain_data(X, y, name='California housing')
    show(exploring_data)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Modelling & uncertainty eval. metrics

    _Regarding the **modelling**: since the focus of this project is **uncertainty estimation** and evaluation metrics, we do not perform any hyperparameter tuning; rather, we choose the default parameters of the algorithms._

    **Metrics**: validity and sharpness are mostly suitable for methods that estimate prediction intervals directly such as quantile and conformal regression. **Validity** has to do with the evaluation of the reliability of quantiles and bias in a probabilistic context. On the other hand, **sharpness** estimates the concentration of probabilities (prediction intervals).

    - **Sharpness**: the average interval width. Sharpness is a measure of how tight the predictive densities are. The narrower the interval length is, the more confident we can be about the prediction.

    _It is also possible that an algorithm is confident about the prediction but is still wrong. How do we measure this? We can count the proportion of real observations that are not within the predictive boundaries of a given interval. The procedure is almost the same as above but instead of calculating the average interval width, we calculate the number of missing observations within the interval._

    - **Validity / Coverage**: for a quantile prediction with some level 0 < α < 1, we expect that (100*α)% of observations are covered. For example, if α is 0.8, we expect that quantile predictions cover 80% of observations. A strong indication for non-valid conditional quantiles is when the empirical quantiles deviate too much from the nominal ones.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Config""")
    return


@app.cell
def _(mo):
    dropdown_alpha = mo.ui.dropdown(
        options=[0.05, 0.1], value=0.05, label="Select alpha: "
    )
    dropdown_alpha
    return (dropdown_alpha,)


@app.cell
def _(dropdown_alpha, norm):
    alpha = dropdown_alpha.value

    # I assume only 0.05 or 0.1 for now
    cov = 90 if alpha==0.1 else 95

    z_score = norm.ppf(1 - alpha/2)
    print(z_score)
    return alpha, cov, z_score


@app.cell
def _(X):
    # Split the data into training, calibration, and testing sets
    n = X.shape[0]
    seed = 0

    #np.random.seed(seed)
    #X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.6, random_state=seed) # 0.7
    #X_calib, X_test, y_calib, y_test = train_test_split(X_temp, y_temp, train_size=1/2, random_state=seed) # 1/3

    # X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=seed)
    return n, seed


@app.cell
def _(X, n, seed, train_test_split, y):
    train_size = 0.6
    calib_size = 0.2
    test_size = 1 - train_size - calib_size

    train_frac = train_size / (train_size + calib_size)
    X_pool, X_test, y_pool, y_test = train_test_split(X, y, test_size=test_size, random_state=42) # fixed test set
    X_train, X_calib, y_train, y_calib = train_test_split(X_pool, y_pool, train_size=train_frac, random_state=seed)


    print(f'Train set dimension: {X_train.shape[0]} ({round(100*X_train.shape[0]/n, 2)}%)')
    print(f'Calibration set dimension: {X_calib.shape[0]} ({round(100*X_calib.shape[0]/n, 2)}%)') # needed for conformal pred
    print(f'Test set dimension: {X_test.shape[0]} ({round(100*X_test.shape[0]/n, 2)}%)')
    return X_calib, X_test, X_train, y_calib, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## LightGBM quantile regression
    LightGBM, a gradient boosting algorithm, is widely used in the machine learning community. Probably, the most straightforward way to get prediction intervals using existing algorithms is to build at least two quantile regression models to target some low and high conditional quantiles. For example, a 90% prediction interval would require fitting two quantile regressions with 5% and 95% quantile levels. **LightGBM implements quantile regression but any other algorithms supporting quantile regression would work too**.

    🔗 https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html
    """
    )
    return


@app.cell
def _(LGBMRegressor, X_test, X_train, y_train):
    # 1. Model to predict expected mean

    # lgb.train is Low level API

    # train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=True) 
    # The free_raw_data parameter controls whether LightGBM keeps the original training data (X_train and y_train) in memory after constructing the internal dataset. When set to True, you cannot access the raw data from the dataset object later.

    lgb_params = {
        'seed': 42,
        'verbose': -1,
        'num_threads': 1,
        'deterministic':True,
        'objective': 'regression',
        #'num_boost_round': 100, # already the default value
        'force_col_wise': True  # force column-wise computation for optimized parallelization
    }
    # lgb_model = lgb.train(params=lgb_params, train_set=train_data) 
    # lgb_mean_pred = lgb_model.predict(X_test)

    # LightGBM uses binning for continuous features to speed up training. This tells you how many bins were created.

    # LGBMRegressor is more scikit-learn style
    lgb_model = LGBMRegressor(**lgb_params)
    lgb_model.fit(X_train, y_train)

    # Predict on train set
    y_train_pred_lgb = lgb_model.predict(X_train)

    # Predict on test set
    y_test_pred_lgb = lgb_model.predict(X_test)
    return lgb_model, y_test_pred_lgb, y_train_pred_lgb


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - **MAE**: when absolute scale matters
    - **MAPE** when relative error matters
    """
    )
    return


@app.cell
def _(mean_squared_error, y_test, y_test_pred_lgb, y_train, y_train_pred_lgb):
    lgb_mse_train = mean_squared_error(y_train, y_train_pred_lgb)
    print(f"MSE on the train set: {round(lgb_mse_train, 3)}")

    lgb_mse_test = mean_squared_error(y_test, y_test_pred_lgb)
    print(f"MSE on the test set: {round(lgb_mse_test, 3)}")
    return


@app.cell
def _(
    mean_absolute_percentage_error,
    y_test,
    y_test_pred_lgb,
    y_train,
    y_train_pred_lgb,
):
    lgb_mape_train = round(mean_absolute_percentage_error(y_train, y_train_pred_lgb)*100, 2)
    print(f"MAPE on the train set: {round(lgb_mape_train, 3)}")

    lgb_mape_test = round(mean_absolute_percentage_error(y_test, y_test_pred_lgb)*100, 2)
    print(f"MAPE on the test set: {round(lgb_mape_test, 3)}")
    return


@app.cell
def _(plt, y_test, y_test_pred_lgb):
    # Residual analysis

    fig_resid, ax_resid = plt.subplots(1, 2, figsize=(10, 4))
    ax_resid[0].plot(y_test, (y_test-y_test_pred_lgb), ls='', marker='o', markersize=3, alpha=0.5)
    ax_resid[0].axhline(0, color='red')
    ax_resid[0].grid()
    ax_resid[0].set_xlabel('y test')
    ax_resid[0].set_ylabel('residuals')

    ax_resid[1].hist((y_test-y_test_pred_lgb), bins=30, edgecolor="black")
    ax_resid[1].grid()
    ax_resid[1].set_xlabel('residuals')

    plt.suptitle('Residual analysis')
    plt.show()
    return


@app.cell
def _(LGBMRegressor, X_calib, X_test, X_train, y_train):
    # Train models on quantiles
    quantiles = [0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.975]

    # Save quantile predictions
    quantile_predictions_test = {}
    quantile_predictions_calib = {}

    for quantile in quantiles:
        print(f"modeling quantile {quantile}")
        lgb_quantile_params = {
            'seed': 42,
            'num_threads': 1,
            'deterministic':True,
            'objective': 'quantile',
            'alpha': quantile,
            'verbose': -1,
            'force_col_wise': True
        }
        lgb_quantile_model = LGBMRegressor(**lgb_quantile_params)
        lgb_quantile_model.fit(X_train, y_train)
        lgb_quantile_calib = lgb_quantile_model.predict(X_calib)
        lgb_quantile_test = lgb_quantile_model.predict(X_test)
        quantile_predictions_calib[quantile] = lgb_quantile_calib
        quantile_predictions_test[quantile] = lgb_quantile_test
    return quantile_predictions_calib, quantile_predictions_test, quantiles


@app.cell
def _(pd, quantile_predictions_test, y_test, y_test_pred_lgb):
    df_lgb_quantile = pd.DataFrame(quantile_predictions_test)
    df_lgb_quantile.columns = ['quantile_' + str(col).split('.')[1] for col in df_lgb_quantile.columns]
    df_lgb_quantile['y_pred'] = y_test_pred_lgb
    df_lgb_quantile['y_true'] = y_test.values
    return (df_lgb_quantile,)


@app.cell
def _(alpha):
    low_quantile, up_quantile = ('quantile_05', 'quantile_95') if alpha==0.1 else ('quantile_025', 'quantile_975')
    return low_quantile, up_quantile


@app.cell
def _(alpha, df_lgb_quantile, low_quantile, plt, up_quantile):
    fig_lgb, ax_lgb = plt.subplots(1, 1)

    ax_lgb.plot(df_lgb_quantile[low_quantile], ls='', marker='o', markersize=2, color='orange', label=f'y q={alpha/2}')
    ax_lgb.plot(df_lgb_quantile['y_pred'], ls='', marker='o', markersize=2, color='red', label='y pred')
    ax_lgb.plot(df_lgb_quantile[up_quantile], ls='', marker='o', markersize=2, color='blue', label=f'y q={1-alpha/2}')

    ax_lgb.grid()
    ax_lgb.legend()

    plt.show()
    return


@app.cell
def _(quantile_predictions_test, quantiles, y_test):
    empirical_quantiles = []
    for q in quantiles:
        empirical = (quantile_predictions_test[q] >= y_test).mean()
        empirical_quantiles.append(empirical)
    return (empirical_quantiles,)


@app.cell
def _(empirical_quantiles, plt, quantiles, sns):
    plt.figure()
    sns.lineplot(x=quantiles, y=quantiles, color="magenta", linestyle='--', linewidth=1, label="ideal")
    sns.lineplot(x=quantiles, y=empirical_quantiles, color="blue", linestyle="dashdot", marker='o', linewidth=1, label="observed")
    # sns.scatterplot(x=quantiles, y=empirical_quantiles, marker="o", s=150)
    plt.legend()
    plt.grid()
    plt.xlabel("True quantile")
    plt.ylabel("Empirical quantile")
    _ = plt.title("Reliability diagram: assessment of quantile predictions \non the test test")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The empirical quantiles (black line) are very close to the ideal quantiles (magenta line) which indicate that quantile predictions generated by LightGBM are valid.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Uncertainty eval.""")
    return


@app.cell
def _(
    calculate_coverage,
    cov,
    df_lgb_quantile,
    low_quantile,
    up_quantile,
    y_test,
):
    # Compute coverage

    lgb_quantile_coverage = calculate_coverage(y_test.values, df_lgb_quantile[low_quantile], df_lgb_quantile[up_quantile])
    print(f"Coverage for LGBoost model with quantile regression ({cov}% coverage interval): {lgb_quantile_coverage*100:.5f}%")
    return (lgb_quantile_coverage,)


@app.cell
def _(
    cov,
    df_lgb_quantile,
    lgb_quantile_coverage,
    low_quantile,
    plot_with_error_bars,
    plt,
    up_quantile,
    y_test,
    y_test_pred_lgb,
):
    fig_lgb_bars, ax_lgb_bars = plt.subplots(1,1)

    plot_with_error_bars(
        y_true=y_test.values,
        y_pred=y_test_pred_lgb,
        lower_bound=df_lgb_quantile[low_quantile], 
        upper_bound=df_lgb_quantile[up_quantile],
        ax=ax_lgb_bars,
        sample_frac=0.1, 
        title=f'Coverage (ideal: {cov}%, nominal: {lgb_quantile_coverage*100:.2f}%)'
    )

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""95% here is too extreme.""")
    return


@app.cell
def _(df_lgb_quantile, pd, plt, sns):
    df_lgb_sharpness = pd.DataFrame(data={
        'cov_95': (df_lgb_quantile['quantile_975']-df_lgb_quantile['quantile_025']),
        'cov_90': (df_lgb_quantile['quantile_95']-df_lgb_quantile['quantile_05']),
        'cov_80': (df_lgb_quantile['quantile_9']-df_lgb_quantile['quantile_1']),
        'cov_70': (df_lgb_quantile['quantile_85']-df_lgb_quantile['quantile_15']),
        'cov_60': (df_lgb_quantile['quantile_8']-df_lgb_quantile['quantile_2'])
    })

    sns.boxplot(data=df_lgb_sharpness)
    plt.grid()
    plt.title('Sharpness at different coverage levels')
    plt.show()
    return (df_lgb_sharpness,)


@app.cell
def _(df_lgb_sharpness):
    df_lgb_sharpness.mean()
    return


@app.cell
def _(X_calib, lgb_model):
    y_calib_pred_lgb = lgb_model.predict(X_calib)
    return (y_calib_pred_lgb,)


@app.cell
def _(
    alpha,
    calculate_coverage,
    get_confidence_interval,
    y_calib,
    y_calib_pred_lgb,
    y_test,
    y_test_pred_lgb,
):
    # Standard CI
    lower_bound_lgb_std, upper_bound_lgb_std = get_confidence_interval(
        y_test_pred=y_test_pred_lgb, 
        y_calib_pred=y_calib_pred_lgb, # it's better to compute std on a new unseen df rather than on the training data
        y_calib=y_calib,
        alpha=alpha
    )

    coverage_lgb_std = calculate_coverage(y_true=y_test, lower_bound=lower_bound_lgb_std, upper_bound=upper_bound_lgb_std)
    print(f"Actual coverage: {round(100*coverage_lgb_std, 2)}%")

    sharpness_lgb_std = (upper_bound_lgb_std-lower_bound_lgb_std).mean()
    print("Actual sharpness:", round(sharpness_lgb_std, 2))
    return


@app.cell
def _(
    alpha,
    calculate_coverage,
    get_conformalized_interval,
    y_calib,
    y_calib_pred_lgb,
    y_test,
    y_test_pred_lgb,
):
    # Conformalized
    lower_bound_lgb_conf, upper_bound_lgb_conf = get_conformalized_interval(
        y_test_pred=y_test_pred_lgb, 
        y_calib_pred=y_calib_pred_lgb,
        y_calib=y_calib,
        alpha=alpha
    )

    coverage_lgb_conf = calculate_coverage(y_true=y_test, lower_bound=lower_bound_lgb_conf, upper_bound=upper_bound_lgb_conf)
    print(f"Actual coverage: {round(100*coverage_lgb_conf, 2)}%")

    sharpness_lgb_conf = (upper_bound_lgb_conf-lower_bound_lgb_conf).mean()
    print("Actual sharpness:", round(sharpness_lgb_conf, 2))
    return


@app.cell
def _(
    low_quantile,
    np,
    pd,
    quantile_predictions_calib,
    up_quantile,
    y_calib,
    y_calib_pred_lgb,
):
    # Conformalized quantile regression (Romano et al.)

    # Conformity scores: how far true y falls outside interval
    # df_lgb_quantile['tmp_col_1'] = df_lgb_quantile[low_quantile] - df_lgb_quantile['y_true']
    # df_lgb_quantile['tmp_col_2'] = df_lgb_quantile['y_true'] - df_lgb_quantile[up_quantile]

    df_lgb_quantile_calib = pd.DataFrame(quantile_predictions_calib)
    df_lgb_quantile_calib.columns = ['quantile_' + str(col).split('.')[1] for col in df_lgb_quantile_calib.columns]
    df_lgb_quantile_calib['y_pred'] = y_calib_pred_lgb
    df_lgb_quantile_calib['y_true'] = y_calib.values

    df_lgb_quantile_calib['q_scores'] = np.maximum(
        df_lgb_quantile_calib[low_quantile] - df_lgb_quantile_calib['y_true'],
        df_lgb_quantile_calib['y_true'] - df_lgb_quantile_calib[up_quantile]
    )

    # df_lgb_quantile_calib
    return (df_lgb_quantile_calib,)


@app.cell
def _(df_lgb_quantile_calib, low_quantile, np):
    random_state = 42
    sample_frac = 0.01 # 20 rows

    def_rng = np.random.default_rng(random_state)
    sampled_idx = def_rng.choice(len(df_lgb_quantile_calib), size=int(len(df_lgb_quantile_calib) * sample_frac), replace=False)
    df_quantile_plot = df_lgb_quantile_calib.iloc[sampled_idx].sort_values(by=low_quantile)
    df_quantile_plot
    return df_quantile_plot, sample_frac


@app.cell
def _(df_quantile_plot, low_quantile, plt, sample_frac, up_quantile):
    # Explanation

    low_q_sample, up_q_sample, y_pred_sample, score_sample, y_true_sample = (
        df_quantile_plot[low_quantile],
        df_quantile_plot[up_quantile],
        df_quantile_plot['y_pred'],
        df_quantile_plot['q_scores'],
        df_quantile_plot['y_true']
    )

    plt.plot(range(len(low_q_sample)), low_q_sample, ls='-', marker='o', markersize=3, color='orange', label='low quantile')
    plt.plot(range(len(y_pred_sample)), y_pred_sample, ls='-', marker='o', markersize=3, color='red', label='pred vals')
    plt.plot(range(len(up_q_sample)), up_q_sample, ls='-', marker='o', markersize=3, color='darkred', label='high quantile')
    plt.plot(range(len(y_true_sample)), y_true_sample, ls='', marker='o', markersize=3, color='blue', label='true y')

    y_end = []
    y_start = []

    for y_val, l, up_q in zip(y_true_sample, score_sample, up_q_sample):
        if y_val > up_q:
            # line goes downward
            y_start.append(y_val)
            y_end.append(y_val - l if l > 0 else y_val)
        else:
            # line goes upward
            y_start.append(y_val)
            y_end.append(y_val + l if l > 0 else y_val)

    plt.vlines(
        range(len(up_q_sample)), 
        y_start,
        y_end,
        color="blue",
        alpha=0.9,
        label='projections'
    )

    plt.legend()
    plt.grid()
    plt.title(f'sample (with frac {sample_frac})')
    plt.show()
    return


@app.cell
def _():
    # [y + l if l > 0 else 0 for y, l in zip(y_true_sample, score_sample)]
    return


@app.cell
def _(alpha, df_lgb_quantile_calib, np):
    q_lgb_conf = np.quantile(df_lgb_quantile_calib['q_scores'], 1 - alpha)
    print(f'conformalized quantile: {q_lgb_conf}')
    return (q_lgb_conf,)


@app.cell
def _(df_lgb_quantile_calib, plt, q_lgb_conf, sns):
    sns.displot(df_lgb_quantile_calib['q_scores'], bins=20, alpha=0.8)
    plt.axvline(0, color='red', label='0')
    plt.axvline(q_lgb_conf, color='blue', label=q_lgb_conf)
    plt.legend()
    plt.grid()
    plt.show()
    # it is doing quite well as most values are negative
    return


@app.cell
def _(
    calculate_coverage,
    df_lgb_quantile,
    low_quantile,
    q_lgb_conf,
    up_quantile,
    y_test,
):
    lower_bound_lgb_conf_quant, upper_bound_lgb_conf_quant = (
        df_lgb_quantile[low_quantile] - q_lgb_conf,
        df_lgb_quantile[up_quantile] + q_lgb_conf
    )

    coverage_lgb_conf_quant = calculate_coverage(
        y_true=y_test.values, 
        lower_bound=lower_bound_lgb_conf_quant.values, 
        upper_bound=upper_bound_lgb_conf_quant.values
    )

    print(f"Actual coverage: {round(100*coverage_lgb_conf_quant, 2)}%")

    sharpness_lgb_conf_quant = (upper_bound_lgb_conf_quant-lower_bound_lgb_conf_quant).mean()
    print("Actual sharpness:", round(sharpness_lgb_conf_quant, 2))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Random forest""")
    return


@app.cell
def _(RandomForestRegressor, X_test, X_train, y_train):
    rf_model = RandomForestRegressor(random_state=42, n_jobs=1)
    rf_model.fit(X_train, y_train)

    # Predict on train set
    y_train_pred_rf = rf_model.predict(X_train)

    # Predict on test set
    y_test_pred_rf= rf_model.predict(X_test)
    return rf_model, y_test_pred_rf, y_train_pred_rf


@app.cell
def _(
    mean_absolute_percentage_error,
    mean_squared_error,
    y_test,
    y_test_pred_rf,
    y_train,
    y_train_pred_rf,
):
    # Model performance

    rf_mse_train = mean_squared_error(y_train, y_train_pred_rf)
    print(f"MSE on the train set: {round(rf_mse_train, 3)}")

    rf_mse_test = mean_squared_error(y_test, y_test_pred_rf)
    print(f"MSE on the test set: {round(rf_mse_test, 10)}")

    rf_mape_train = round(mean_absolute_percentage_error(y_train, y_train_pred_rf)*100, 2)
    print(f"MAPE on the train set: {round(rf_mape_train, 3)}")

    rf_mape_test = round(mean_absolute_percentage_error(y_test, y_test_pred_rf)*100, 2)
    print(f"MAPE on the test set: {round(rf_mape_test, 3)}")
    return


@app.cell
def _(plt, y_test, y_test_pred_rf, y_train, y_train_pred_rf):
    fig_rf_resid, ax_rf_resid = plt.subplots(1, 2, figsize=(10, 4))

    ax_rf_resid[0].plot(y_train, (y_train-y_train_pred_rf), ls='', marker='o', markersize=3, alpha=0.5)
    ax_rf_resid[0].axhline(0, color='red')
    ax_rf_resid[0].grid()
    ax_rf_resid[0].set_title('train set')
    ax_rf_resid[0].set_xlabel('y train')
    ax_rf_resid[0].set_ylabel('residuals')

    ax_rf_resid[1].plot(y_test, (y_test-y_test_pred_rf), ls='', marker='o', markersize=3, alpha=0.5)
    ax_rf_resid[1].axhline(0, color='red')
    ax_rf_resid[1].grid()
    ax_rf_resid[1].set_title('test set')
    ax_rf_resid[1].set_xlabel('y test')
    ax_rf_resid[1].set_ylabel('residuals')

    plt.suptitle('Residuals')
    plt.show()
    return


@app.cell
def _(RandomForestQuantileRegressor, X_calib, X_test, X_train, y_train):
    # Quantile regression
    qrf = RandomForestQuantileRegressor(random_state=42)
    qrf.fit(X_train, y_train)

    y_test_pred_qrf = qrf.predict(X_test, quantiles=[0.025, 0.975])
    y_calib_pred_qrf = qrf.predict(X_calib, quantiles=[0.025, 0.975])
    return y_calib_pred_qrf, y_test_pred_qrf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Uncertainty eval.""")
    return


@app.cell
def _(X_calib, rf_model):
    y_calib_pred_rf = rf_model.predict(X_calib)
    return (y_calib_pred_rf,)


@app.cell
def _(
    alpha,
    get_confidence_interval,
    y_calib,
    y_calib_pred_rf,
    y_test_pred_rf,
):
    # Method 1
    # Standard confidence intervals
    lower_bound_rf_std, upper_bound_rf_std = get_confidence_interval(
        y_test_pred=y_test_pred_rf, 
        y_calib_pred=y_calib_pred_rf,
        y_calib=y_calib,
        alpha=alpha
    )
    return lower_bound_rf_std, upper_bound_rf_std


@app.cell
def _(calculate_coverage, lower_bound_rf_std, upper_bound_rf_std, y_test):
    rf_std_coverage = calculate_coverage(y_true=y_test, lower_bound=lower_bound_rf_std, upper_bound=upper_bound_rf_std)
    print(f"Actual coverage: {round(100*rf_std_coverage, 2)}%")

    sharpness_rf_std = (upper_bound_rf_std-lower_bound_rf_std).mean()
    print("Actual sharpness:", round(sharpness_rf_std, 2))
    return (rf_std_coverage,)


@app.cell
def _(X_test, np, rf_model):
    # Method 2: uncertainty from trees
    rf_all_tree_preds = np.stack([tree.predict(np.array(X_test)) for tree in rf_model.estimators_], axis=0)
    rf_preds = rf_all_tree_preds.mean(axis=0) # same as y_pred for test set with rf
    rf_uncertainties = rf_all_tree_preds.std(axis=0)
    rf_uncertainties
    return rf_preds, rf_uncertainties


@app.cell
def _(calculate_coverage, cov, rf_preds, rf_uncertainties, y_test, z_score):
    # Compute coverage
    lower_bound_rf = rf_preds - z_score * rf_uncertainties
    upper_bound_rf = rf_preds + z_score * rf_uncertainties

    rf_quantile_coverage = calculate_coverage(y_test.values, lower_bound_rf, upper_bound_rf)
    print(f"Coverage for RF model ({cov}% coverage interval): {rf_quantile_coverage*100:.2f}%")

    rf_sharpness = round((upper_bound_rf-lower_bound_rf).mean(), 2)
    print("Sharpness:", rf_sharpness) 
    return lower_bound_rf, rf_quantile_coverage, upper_bound_rf


@app.cell
def _(
    cov,
    lower_bound_rf,
    lower_bound_rf_std,
    plot_with_error_bars,
    plt,
    rf_quantile_coverage,
    rf_std_coverage,
    upper_bound_rf,
    upper_bound_rf_std,
    y_test,
    y_test_pred_rf,
):
    fig_rf, ax_rf= plt.subplots(1, 2, figsize=(10, 4))

    plot_with_error_bars(
        y_true=y_test.values,
        y_pred=y_test_pred_rf,
        lower_bound=lower_bound_rf, 
        upper_bound=upper_bound_rf,
        ax=ax_rf[0],
        sample_frac=0.1, 
        title=f'Built-in method\nCoverage (ideal={cov}%, nominal={round(100*rf_quantile_coverage, 2)}%)'
    )

    plot_with_error_bars(
        y_true=y_test.values,
        y_pred=y_test_pred_rf,
        lower_bound=lower_bound_rf_std, 
        upper_bound=upper_bound_rf_std,
        ax=ax_rf[1],
        sample_frac=0.1, 
        title=f'Standard CI\nCoverage (ideal={cov}%, nominal={round(100*rf_std_coverage, 2)}%)'
    )

    plt.show()
    return


@app.cell
def _(
    alpha,
    calculate_coverage,
    cov,
    get_conformalized_interval,
    y_calib,
    y_calib_pred_rf,
    y_test,
    y_test_pred_rf,
):
    conf_lower_bound_rf, conf_upper_bound_rf = get_conformalized_interval(
        y_test_pred=y_test_pred_rf, 
        y_calib_pred=y_calib_pred_rf,
        y_calib=y_calib,
        alpha=alpha
    )

    conf_rf_quantile_coverage = calculate_coverage(y_test.values, conf_lower_bound_rf, conf_upper_bound_rf)
    print(f"Coverage for RF model with quantile regression ({cov}% coverage interval): {conf_rf_quantile_coverage*100:.5f}%")

    sharpness_conf_rf = (conf_upper_bound_rf-conf_lower_bound_rf).mean()
    print("Actual sharpness:", round(sharpness_conf_rf, 2))
    return


@app.cell
def _(calculate_coverage, cov, y_test, y_test_pred_qrf):
    # Quantile regression
    qrf_quantile_coverage = calculate_coverage(y_test.values, y_test_pred_qrf[:, 0], y_test_pred_qrf[:, 1])
    print(f"Coverage for RF model with quantile regression ({cov}% coverage interval): {qrf_quantile_coverage*100:.5f}%")

    sharpness_qrf = (y_test_pred_qrf[:, 1]-y_test_pred_qrf[:, 0]).mean()
    print("Actual sharpness:", round(sharpness_qrf, 2))
    return


@app.cell
def _(
    alpha,
    calculate_coverage,
    cov,
    np,
    y_calib,
    y_calib_pred_qrf,
    y_test,
    y_test_pred_qrf,
):
    # Conformal quantile regression
    qrf_uncertainty_score = np.maximum(
        y_calib_pred_qrf[:, 0] - y_calib,
        y_calib - y_calib_pred_qrf[:, 1]
    )
    qrf_quantile_score = np.quantile(qrf_uncertainty_score, 1 - alpha)
    conf_quant_lower_bound, conf_quant_upper_bound = (
        y_test_pred_qrf[:, 0] - qrf_quantile_score,
        y_test_pred_qrf[:, 1] + qrf_quantile_score
    )

    quant_qrf_quantile_coverage = calculate_coverage(y_test.values, conf_quant_lower_bound, conf_quant_upper_bound)
    print(f"Coverage for conf QRF model ({cov}% coverage interval): {quant_qrf_quantile_coverage*100:.5f}%")

    sharpness_quant_qrf = (conf_quant_upper_bound-conf_quant_lower_bound).mean()
    print("Actual sharpness:", round(sharpness_quant_qrf, 2))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## EBM
    **EBMs** (Explainable Boosting Machines) are **bagged generalized additive models (GAMs)**. They train multiple bagged models, each on a bootstrap sample of the training data. Each bagged model is independent of the others because it sees different samples.
    Within a single bagged model:

    - Each bagged model is itself trained iteratively on each feature using boosting-like updates:
        - For feature 1, learn a small function that explains the target.
        - Move to feature 2, fit a function on the residuals, etc.
    - These iterations inside a bagged model are not independent — they are sequential updates to reduce residual error.

    `predict_with_uncertainty` then:

    - collects predictions from all bagged models,
    - computes the mean prediction and standard deviation across them.

    **✅ Key point:**

    - Iterations within one EBM → sequential, not independent.
    - Different bagged EBMs → independent because each uses a bootstrap sample of the training data.
    - Uncertainty estimate comes from the spread across these independent bagged models.
    """
    )
    return


@app.cell
def _(ExplainableBoostingRegressor, X_train, y_train):
    # Initialize and train the Explainable Boosting Machine (EBM) Regressor
    ebm_model = ExplainableBoostingRegressor(random_state=42)
    ebm_model.fit(X_train, y_train)
    return (ebm_model,)


@app.cell
def _(X_test, X_train, ebm_model):
    # Predictions
    y_train_pred_ebm = ebm_model.predict(X_train)
    y_test_pred_ebm = ebm_model.predict(X_test)
    return y_test_pred_ebm, y_train_pred_ebm


@app.cell
def _(model_eval_metrics, y_test, y_test_pred_ebm, y_train, y_train_pred_ebm):
    train_metrics = model_eval_metrics(y_train, y_train_pred_ebm)
    test_metrics = model_eval_metrics(y_test, y_test_pred_ebm)

    ebm_mse_train, ebm_mape_train = train_metrics['mse'], train_metrics['mape']
    ebm_mse_test, ebm_mape_test = test_metrics['mse'], test_metrics['mape']
    return ebm_mape_test, ebm_mape_train, ebm_mse_test, ebm_mse_train


@app.cell
def _(ebm_mape_test, ebm_mape_train, ebm_mse_test, ebm_mse_train):
    print(f"MSE on the training set: {round(ebm_mse_train, 3)}")
    print(f"MSE on the test set: {round(ebm_mse_test, 3)}")

    print(f"MAPE on the training set: {round(ebm_mape_train, 3)}%")
    print(f"MAPE on the test set: {round(ebm_mape_test, 3)}%")
    return


@app.cell
def _(
    ebm_mse_test,
    ebm_mse_train,
    plt,
    true_vs_pred,
    y_test,
    y_test_pred_ebm,
    y_train,
    y_train_pred_ebm,
):
    fig_ebm_res, ax_ebm_res = plt.subplots(1, 2, figsize=(12, 5))

    true_vs_pred(y_true=y_train, y_pred=y_train_pred_ebm, mse=ebm_mse_train, ax=ax_ebm_res[0])
    true_vs_pred(y_true=y_test, y_pred=y_test_pred_ebm, mse=ebm_mse_test, ax=ax_ebm_res[1])

    plt.suptitle('Predicted vs true values')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Uncertainty eval.""")
    return


@app.cell
def _(X_calib, ebm_model):
    y_calib_pred_ebm = ebm_model.predict(X_calib)
    return (y_calib_pred_ebm,)


@app.cell
def _(X_test, ebm_model):
    # Method 1
    preds, uncertainties = ebm_model.predict_with_uncertainty(X_test).T
    preds, uncertainties

    # Uncertainty estimates → usually the standard deviation (or variance) of the prediction, based on the distribution of trees in the boosted ensemble.
    # uncertainties → per-sample uncertainty from disagreement among bags
    # This works because the models are independent, so the spread of predictions reflects model uncertainty.

    # EBMs are additive models trained with bagging:
    # For each bag (base model), a bootstrap sample of the training data is used.
    # Each base model learns its own additive functions on that bootstrap sample.
    # Because each bag sees slightly different data, the resulting models are independent draws from the training distribution.
    return preds, uncertainties


@app.cell
def _(calculate_coverage, cov, preds, uncertainties, y_test, z_score):
    # Compute coverage
    lower_bound_ebm = preds - z_score * uncertainties
    upper_bound_ebm = preds + z_score * uncertainties

    ebm_quantile_coverage = calculate_coverage(y_test.values, lower_bound_ebm, upper_bound_ebm)
    print(f"Coverage for EBM model ({cov}% coverage interval): {ebm_quantile_coverage*100:.2f}%")

    ebm_sharpness = round((upper_bound_ebm-lower_bound_ebm).mean(), 2)
    print("Sharpness:", ebm_sharpness) 
    return ebm_quantile_coverage, lower_bound_ebm, upper_bound_ebm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""**Key takeaway**: EBMs’ `predict_with_uncertainty` **does not guarantee coverage**. The numeric value is just the model’s internal uncertainty estimate, which can be too small —especially for extreme values or complex distributions.""")
    return


@app.cell
def _(calculate_coverage):
    calculate_coverage
    return


@app.cell
def _(
    alpha,
    calculate_coverage,
    get_confidence_interval,
    y_calib,
    y_calib_pred_ebm,
    y_test,
    y_test_pred_ebm,
):
    # Method 2
    # Standard confidence intervals
    lower_bound_ebm_std, upper_bound_ebm_std = get_confidence_interval(
        y_test_pred=y_test_pred_ebm, 
        y_calib_pred=y_calib_pred_ebm,
        y_calib=y_calib,
        alpha=alpha)

    coverage_ebm_std = calculate_coverage(y_true=y_test, lower_bound=lower_bound_ebm_std, upper_bound=upper_bound_ebm_std)
    print(f"Actual coverage: {round(100*coverage_ebm_std, 5)}%")

    sharpness_ebm_std = (upper_bound_ebm_std-lower_bound_ebm_std).mean()
    print("Actual sharpness:", round(sharpness_ebm_std, 2))
    return coverage_ebm_std, lower_bound_ebm_std, upper_bound_ebm_std


@app.cell
def _(
    alpha,
    calculate_coverage,
    get_conformalized_interval,
    y_calib,
    y_calib_pred_ebm,
    y_test,
    y_test_pred_ebm,
):
    # Method 3: conformalized prediction intervals
    # This method from conformal prediction gives prediction intervals with guaranteed coverage (in this case, ~95% if alpha=0.05), without making strong distributional assumptions.

    # Conformal prediction uses calibration data to “correct” the model by wrapping intervals around predictions.

    lower_bound_ebm_conf, upper_bound_ebm_conf = get_conformalized_interval(
        y_test_pred=y_test_pred_ebm, 
        y_calib_pred=y_calib_pred_ebm,
        y_calib=y_calib,
        alpha=alpha
    )

    coverage_ebm_conf = calculate_coverage(y_true=y_test, lower_bound=lower_bound_ebm_conf, upper_bound=upper_bound_ebm_conf)
    print(f"Actual coverage: {round(100*coverage_ebm_conf, 6)}%")

    sharpness_ebm_conf = (upper_bound_ebm_conf-lower_bound_ebm_conf).mean()
    print("Actual sharpness:", round(sharpness_ebm_conf, 2))
    return (
        coverage_ebm_conf,
        lower_bound_ebm_conf,
        sharpness_ebm_conf,
        upper_bound_ebm_conf,
    )


@app.cell
def _(
    cov,
    coverage_ebm_conf,
    coverage_ebm_std,
    ebm_quantile_coverage,
    lower_bound_ebm,
    lower_bound_ebm_conf,
    lower_bound_ebm_std,
    plot_with_error_bars,
    plt,
    upper_bound_ebm,
    upper_bound_ebm_conf,
    upper_bound_ebm_std,
    y_test,
    y_test_pred_ebm,
):
    fig_ebm, ax_ebm = plt.subplots(1, 3, figsize=(13, 4))

    plot_with_error_bars(
        y_true=y_test.values,
        y_pred=y_test_pred_ebm,
        lower_bound=lower_bound_ebm, 
        upper_bound=upper_bound_ebm,
        ax=ax_ebm[0],
        sample_frac=0.1, 
        title=f'Built-in method\nCoverage (ideal={cov}%, nominal={round(100*ebm_quantile_coverage, 2)}%)'
    )

    plot_with_error_bars(
        y_true=y_test.values,
        y_pred=y_test_pred_ebm,
        lower_bound=lower_bound_ebm_std, 
        upper_bound=upper_bound_ebm_std,
        ax=ax_ebm[1],
        sample_frac=0.1, 
        title=f'Standard CI\nCoverage (ideal={cov}%, nominal={round(100*coverage_ebm_std, 2)}%)'
    )

    plot_with_error_bars(
        y_true=y_test.values,
        y_pred=y_test_pred_ebm,
        lower_bound=lower_bound_ebm_conf, 
        upper_bound=upper_bound_ebm_conf,
        ax=ax_ebm[2],
        sample_frac=0.1, 
        title=f'Conformalized intervals\nCoverage (ideal={cov}%, nominal={round(100*coverage_ebm_conf, 2)}%)'
    )

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **Notes**

    When talking about the “built-in interval” in RF and EBM, these are not true theoretical confidence intervals, but rather a **heuristic estimate based on the variance of the base models’ predictions (the trees)**.

    - Random Forest (RF): the variance across trees tends to capture both aleatory uncertainty (noise in the data) and model uncertainty (how predictions change across bootstrap samples). This often produces relatively wide empirical intervals, resulting in higher coverage.

    - Explainable Boosting Machine (EBM): here, the “prediction std” has a different meaning: EBM does not use bagging like RF but instead additive boosting. The variance across iterations does not directly reflect data uncertainty, but rather the stability of the boosting process itself. As a result, the intervals calculated this way are often far too narrow → which explains why you observed very low coverage (21% vs. a nominal 90%).

    ```
    In summary:

    - In RF, the dispersion across trees is a reasonable proxy for uncertainty.

    - In EBM, it is not → intervals are not calibrated.
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## XGBoost model""")
    return


@app.cell
def _(XGBRegressor):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 150
    }

    # Initialize the model
    xgb_model = XGBRegressor(**params)
    return (xgb_model,)


@app.cell
def _():
    # Define hyperparameters to test
    param_grid = {
        'max_depth': [5, 6, 10],
        'learning_rate': [0.03, 0.05, 0.1],
        'n_estimators': [50, 100],
        'subsample': [0.75, 1], 
        'colsample_bytree': [0.75, 1]  
    }

    total_combinations = 1
    for values in param_grid.values():
        total_combinations *= len(values)

    print(f'Total number of parameter combinations: {total_combinations}')
    return (param_grid,)


@app.cell
def _(GridSearchCV, param_grid, xgb_model):
    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=3
    )

    # Train the model with different hyperparameter combinations
    #grid_search.fit(X_train, y_train)
    return (grid_search,)


@app.cell
def _(grid_search):
    # Display the best parameters
    best_params = grid_search.best_params_
    print("Best Parameters:", best_params)

    # Use the model with the best parameters
    best_model = grid_search.best_estimator_
    return (best_model,)


@app.cell
def _(XGBRegressor, X_train, y_train):
    # default model
    xgb_model_def = XGBRegressor()
    xgb_model_def.fit(X_train, y_train)
    return (xgb_model_def,)


@app.cell(hide_code=True)
def _(mo):
    dropdown_xgboost_model = mo.ui.dropdown(
        options=['Simple (default) model', 'Best model (CV grid search optimized)'], value='Simple (default) model', label="Select the model: "
    )
    dropdown_xgboost_model
    return (dropdown_xgboost_model,)


@app.cell
def _(
    X_test,
    X_train,
    best_model,
    dropdown_xgboost_model,
    mean_squared_error,
    xgb_model_def,
    y_test,
    y_train,
):
    # Evaluate the model on the training set 
    xgb_model_sel = xgb_model_def if dropdown_xgboost_model.value == 'Simple (default) model' else best_model

    y_train_pred = xgb_model_sel.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_pred)
    print(f"MSE on the training set: {round(mse_train, 3)}")

    # Evaluate the model on the test set 
    y_test_pred = xgb_model_sel.predict(X_test)
    mse_test = mean_squared_error(y_test, y_test_pred)
    print(f"MSE on the test set: {round(mse_test, 3)}")
    return mse_test, mse_train, xgb_model_sel, y_test_pred, y_train_pred


@app.cell
def _(
    mse_test,
    mse_train,
    plt,
    true_vs_pred,
    y_test,
    y_test_pred,
    y_train,
    y_train_pred,
):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    true_vs_pred(y_true=y_train, y_pred=y_train_pred, mse=mse_train, ax=ax[0])
    true_vs_pred(y_true=y_test, y_pred=y_test_pred, mse=mse_test, ax=ax[1])

    plt.suptitle('Predicted vs True values')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The **threshold-effect for high-valued houses** stands out: all houses with a price above 5 are given the value 5.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Feature importance""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **Score choice**:

    - `weight` is the count of times a feature is used in splits,
    - `gain` is the average gain from those splits,
    - `cover` is the average coverage of samples at those splits,
    - `total_gain` is the sum of gains across all splits,
    - `total_cover` is the sum of coverages across all splits.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    dropdown_ft_imp_score = mo.ui.dropdown(
        options=['weight', 'gain', 'cover', 'total_gain', 'total_cover'], value='weight', label="Select the feature importance score: "
    )
    dropdown_ft_imp_score
    return (dropdown_ft_imp_score,)


@app.cell
def _(dropdown_ft_imp_score, pd, xgb_model_sel):
    ft_imp_score = dropdown_ft_imp_score.value
    ft_imp_values = xgb_model_sel.get_booster().get_score(importance_type=ft_imp_score)

    df_ft_imp = pd.DataFrame(
        data=list(ft_imp_values.values()), 
        index=list(ft_imp_values.keys()), 
        columns=["score"]
    ).sort_values(by="score", ascending=True)
    return df_ft_imp, ft_imp_score


@app.cell
def _(df_ft_imp, ft_imp_score, plt):
    fig_ft_imp, ax_ft_imp = plt.subplots()
    ax_ft_imp.barh(range(len(df_ft_imp)), df_ft_imp.score, color='green', alpha=0.5)
    ax_ft_imp.set_yticks(range(len(df_ft_imp)), df_ft_imp.index)
    ax_ft_imp.grid()
    ax_ft_imp.set_title(f'Feature importance based on {ft_imp_score}')
    plt.show()
    return


@app.cell
def _(X_test, boston, permutation_importance, plt, xgb, y_test):
    perm_importance = permutation_importance(xgb, X_test, y_test)
    sorted_idx = perm_importance.importances_mean.argsort()
    plt.barh(boston.feature_names[sorted_idx], perm_importance.importances_mean[sorted_idx])
    plt.xlabel("Permutation Importance")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Shap explainability""")
    return


@app.cell
def _(X_test, shap, xgb_model_sel):
    explainer = shap.TreeExplainer(xgb_model_sel)
    shap_values = explainer.shap_values(X_test)
    return (shap_values,)


@app.cell
def _(X_test, shap, shap_values):
    shap.summary_plot(shap_values, X_test, plot_type="bar")
    return


@app.cell(column=1)
def _():
    import crepes
    import mapie

    from crepes import WrapRegressor
    return (WrapRegressor,)


@app.cell
def _(coverage_ebm_conf, sharpness_ebm_conf):
    # From the previous column...

    # y_train_pred_ebm
    # y_calib_pred_ebm
    # y_test_pred_ebm

    # calculate_coverage(y_true=y_test, lower_bound=lower_bound_ebm_conf, upper_bound=upper_bound_ebm_conf)
    print(f"Actual coverage: {round(100*coverage_ebm_conf, 2)}%")

    # sharpness_ebm_conf = (upper_bound_ebm_conf-lower_bound_ebm_conf).mean()
    print("Actual sharpness:", round(sharpness_ebm_conf, 2))
    return


@app.cell
def _(
    calculate_coverage,
    np,
    y_calib,
    y_calib_pred_ebm,
    y_test,
    y_test_pred_ebm,
):
    # INGE COMPUTATION

    confidence = 0.95 
    residuals = y_calib - y_calib_pred_ebm
    conformity_scores = np.sort(np.abs(residuals))[::-1]
    quantile_index = int((1 - confidence) * (len(conformity_scores) + 1)) - 1
    width = conformity_scores[quantile_index]

    prediction_intervals = np.zeros((len(y_test_pred_ebm), 2))
    prediction_intervals[:, 0] = y_test_pred_ebm - width
    prediction_intervals[:, 1] = y_test_pred_ebm + width

    calculate_coverage(y_true=y_test, lower_bound=prediction_intervals[:, 0], upper_bound=prediction_intervals[:, 1])
    return confidence, prediction_intervals


@app.cell
def _(lower_bound_ebm_conf, pd, prediction_intervals, upper_bound_ebm_conf):
    df_ebm = pd.DataFrame(prediction_intervals, columns=['manual_lb_inge', 'manual_ub_inge'])
    df_ebm['manual_lb'] = lower_bound_ebm_conf
    df_ebm['manual_ub'] = upper_bound_ebm_conf

    df_ebm
    return


@app.cell
def _(lower_bound_ebm_conf, upper_bound_ebm_conf):
    lower_bound_ebm_conf, upper_bound_ebm_conf
    return


@app.cell
def _(
    WrapRegressor,
    X_calib,
    X_test,
    X_train,
    confidence,
    ebm_model,
    y_calib,
    y_train,
):
    crepes_model = WrapRegressor(ebm_model)
    crepes_model.fit(X_train, y_train)
    crepes_model.calibrate(X_calib, y_calib)
    crepes_point_prediction = crepes_model.predict(X_test)
    crepes_prediction_intervals = crepes_model.predict_int(X_test, confidence=confidence)
    return (crepes_prediction_intervals,)


@app.cell
def _(crepes_prediction_intervals):
    crepes_prediction_intervals
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
