import os
import glob
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import calplot
import matplotlib.pyplot as plt

SMALL_SIZE = 8
MEDIUM_SIZE = 12
LARGE_SIZE = 16


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


def calplot_values_per_day(df, col, title="", filename=""):
    daily_counts = df[col].resample("D").count()
    daily_counts = daily_counts.tz_localize(None)
    fig, ax = calplot.calplot(
        daily_counts,
        # cmap="jet",
        figsize=(12, 4),
        colorbar=True,
        yearlabel_kws={"fontname": "sans-serif"},
        suptitle=title,
    )
    plt.show()


def clean_IQR(df, test="GTI_mV", ref="GHI_mV"):
    df = df.dropna()
    resids = df[test] - df[ref]
    q75 = np.percentile(resids, 75)
    q25 = np.percentile(resids, 25)
    iqr = q75 - q25  # InterQuantileRange
    is_good = (resids > (q25 - 1.5 * iqr)) & (resids < (q75 + 1.5 * iqr))
    good = df[is_good]
    outliers = df[~is_good]
    resids = good[test] - good[ref]
    mbe = np.mean(resids)
    rmse = mean_squared_error(good[ref], good[test])
    iqr_dict = {"mbe": mbe, "rmse": rmse}
    STATS_DICT.update(iqr_dict)
    return good, outliers


def calc_linregress(x, y):
    model = sm.OLS(y, x)
    results = model.fit()
    return results


def plot_scatter(df, x, y, filename="", title=""):
    plt.rc("font", size=MEDIUM_SIZE)
    plt.scatter(df[x], df[y], edgecolor="k")
    # plt.xlim(-3,7)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def save_regresults(
    df, results, good, outliers, ref_col="Reference", fname="out/stats.json"
):
    linregress_dict = {
        "slope": results.params[ref_col],
        "slope_stderr": results.bse.values[0],
        "r2": results.rsquared,
        "pvalue": results.pvalues.values[0],
        "count": len(df),
        "outliers": len(outliers),
        "non-outliers": len(good),
    }
    STATS_DICT.update(linregress_dict)
    print(json.dumps(STATS_DICT, indent=4))
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, "w") as f:
        json.dump(STATS_DICT, f, indent=4)


def plot_regression(good, outliers, results, filename=""):
    x, y = good[ref_col], good[test_col]
    fig, ax = plt.subplots()
    ax.scatter(x, y, s=14)
    ax.scatter(outliers[ref_col], outliers[test_col], s=14, label="Outliers")
    slope = results.params[ref_col]
    slope_stderr = results.bse.values[0]
    r2 = results.rsquared
    label = f"$y={slope:.5f}x \pm {slope_stderr:.1},\ R^2={r2:.4f}$"
    ax.plot(x, slope * x, color="red", label=label)
    ax.legend()
    ax.set_xlabel(f"Reference: {ref_col} (W/m2)")
    ax.set_ylabel(f"Test: {test_col} (W/m2)")
    ax.set_title(x.index[0].date())
    if filename:
        plt.savefig(filename)
    plt.show()


df_all = load_dataset()

calplot_values_per_day(df_all, "GHI_pred", title="GHI predicted values per day")
calplot_values_per_day(df_all, "GHI_meas", title="GHI measured values per day")

plot_scatter(df_all, "GHI_meas", "GHI_pred")
plot_scatter(df_all, "DHI_meas", "DHI_pred")


ref_col = "GHI_meas"
test_col = "GHI_pred"

date_and_cols = f"{df_all.index[0].date()}_{ref_col}_{test_col}"

STATS_DICT = {
    "start": f"{df_all.index[0]}",
    "end": f"{df_all.index[-1]}",
    "ref_col": ref_col,
    "test_col": test_col,
}

good, outliers = clean_IQR(df_all, test=test_col, ref=ref_col)

reg_results = calc_linregress(good[ref_col], good[test_col])
save_regresults(
    df_all,
    reg_results,
    good,
    outliers,
    ref_col=ref_col,
    fname=f"out/stats_{date_and_cols}.json",
)

plot_regression(
    good, outliers, reg_results, filename=f"out/regression_{date_and_cols}.png"
)
