import os
import calplot
import matplotlib.pyplot as plt

SMALL_SIZE = 8
MEDIUM_SIZE = 12
LARGE_SIZE = 16


def savefig(filename):
    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)


def calplot_values_per_day(df, col, title="", filename=""):
    plt.rc("font", size=MEDIUM_SIZE)
    daily_counts = df[col].resample("D").count()
    daily_counts = daily_counts.tz_localize(None)
    fig, ax = calplot.calplot(
        daily_counts,
        # cmap="jet",
        figsize=(12, 4),
        colorbar=True,
        yearlabel_kws={"fontname": "sans-serif"},
    )
    plt.suptitle(title)
    savefig(filename)
    plt.show()


def plot_hist(df, ref_col, test_col, filename=""):
    plt.rc("font", size=MEDIUM_SIZE)
    plt.hist(df[ref_col], bins=30, label=ref_col, edgecolor="k", alpha=0.7)
    plt.hist(df[test_col], bins=30, label=test_col, edgecolor="k", alpha=0.7)
    plt.xlabel(f"{ref_col[:3]} (W/m2)")
    plt.ylabel("Count")
    plt.legend()
    savefig(filename)
    plt.show()


def plot_regression(df, results, ref_col, test_col, filename=""):
    good = df[~df["is_outlier"]]
    outliers = df[df["is_outlier"]]
    x, y = good[ref_col], good[test_col]

    slope = results.params[ref_col]
    slope_stderr = results.bse.values[0]
    r2 = results.rsquared
    label = f"$y={slope:.4f}x \pm {slope_stderr:.1},\ R^2={r2:.2f}$"

    plt.rc("font", size=MEDIUM_SIZE)
    fig, ax = plt.subplots()
    ax.scatter(x, y, s=14, edgecolor="k", linewidths=0.4)
    ax.scatter(
        outliers[ref_col],
        outliers[test_col],
        s=14,
        linewidths=0.4,
        label="Outliers",
        edgecolor="k",
    )
    ax.plot(x, slope * x, color="red", label=label)
    ax.legend()
    ax.set_xlabel(f"Reference: {ref_col} (W/m2)")
    ax.set_ylabel(f"Test: {test_col} (W/m2)")
    savefig(filename)
    plt.show()
