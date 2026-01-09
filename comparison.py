import glob
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from plotting import calplot_values_per_day, plot_regression, plot_hist
from report import ModelReport


def read_csv_files(files):
    df_list = []
    for fname in files:
        df = pd.read_csv(fname, parse_dates=True, index_col="Datetime_UTC")
        df_list.append(df)
    df_all = pd.concat(df_list)
    return df_all


def load_dataset():
    prediction_files = glob.glob("predictions/**/*.csv", recursive=True)
    solar_files = glob.glob("solar1min/**/*.csv", recursive=True)

    predictions = read_csv_files(prediction_files)
    predictions = predictions.rename(
        columns={"GHI": "GHI_pred", "DHI": "DHI_pred"}
    )
    predictions = predictions.resample("1min").first()

    solar = read_csv_files(solar_files)
    solar = solar[["GHI_Avg", "DHI_Avg"]]
    solar = solar.rename(columns={"GHI_Avg": "GHI_meas", "DHI_Avg": "DHI_meas"})
    solar = solar.tz_localize("UTC")

    return pd.concat([solar, predictions], axis=1)


def flag_iqr_outliers(df, test="GHI_mV", ref="GHI_mV", multiplier=1.5):
    df = df.copy()
    residuals = df[test] - df[ref]
    q25 = residuals.quantile(0.25)
    q75 = residuals.quantile(0.75)
    iqr = q75 - q25
    lower_bound = q25 - multiplier * iqr
    upper_bound = q75 + multiplier * iqr
    df["is_outlier"] = ~residuals.between(lower_bound, upper_bound)
    return df


def calc_linregress(x, y):
    model = sm.OLS(y, x)
    results = model.fit()
    return results


def calc_performance(ref, test):
    mae = mean_absolute_error(ref, test)
    rmse = root_mean_squared_error(ref, test)
    mbe = np.mean(test - ref)
    return mae, rmse, mbe


def run_analysis(df, ref_col, test_col):
    calplot_values_per_day(
        df,
        "GHI_pred",
        title="GHI predicted values per day",
        filename="out/calplot_GHI_pred.png",
    )
    calplot_values_per_day(
        df,
        "GHI_meas",
        title="GHI measured values per day",
        filename="out/calplot_GHI_meas.png",
    )

    df = df.dropna()
    df = flag_iqr_outliers(df, test=test_col, ref=ref_col)
    clean_df = df[~df["is_outlier"]]
    results = calc_linregress(clean_df[ref_col], clean_df[test_col])
    mae, rmse, mbe = calc_performance(clean_df[ref_col], clean_df[test_col])

    report = ModelReport(
        start=f"{df.index[0]}",
        end=f"{df.index[-1]}",
        ref_col=ref_col,
        test_col=test_col,
        all_values=len(df),
        outliers=len(df) - len(clean_df),
        slope=results.params.iloc[0],
        slope_err=results.bse.iloc[0],
        r2=results.rsquared,
        pvalue=results.pvalues.iloc[0],
        nobs=int(results.nobs),
        mae=mae,
        rmse=rmse,
        mbe=mbe,
    )
    plot_hist(df, ref_col, test_col, filename=f"out/hist_{ref_col}_{test_col}")
    plot_regression(
        df,
        results,
        ref_col,
        test_col,
        filename=f"out/regression_{ref_col}_{test_col}",
    )
    report.to_json(f"out/report_{ref_col}_{test_col}.json")


df = load_dataset()
run_analysis(df, "GHI_meas", "GHI_pred")
run_analysis(df, "DHI_meas", "DHI_pred")
