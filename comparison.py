import glob
import pandas as pd
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


def plot_scatter(df, x, y, filename="", title=""):
    plt.rc("font", size=MEDIUM_SIZE)
    plt.scatter(df[x], df[y], edgecolor="k")
    # plt.xlim(-3,7)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.tight_layout()
    plt.show()


df_all = load_dataset()

calplot_values_per_day(df_all, "GHI_pred", title="GHI predicted values per day")
calplot_values_per_day(df_all, "GHI_meas", title="GHI measured values per day")

plot_scatter(df_all, "GHI_meas", "GHI_pred")
plot_scatter(df_all, "DHI_meas", "DHI_pred")
