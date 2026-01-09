from dataclasses import dataclass, asdict
import json


@dataclass(frozen=True)
class ModelReport:
    start: str
    end: str
    ref_col: str
    test_col: str
    all_values: int
    outliers: int
    slope: float
    slope_err: float
    r2: float
    pvalue: float
    nobs: int
    mae: float
    rmse: float
    mbe: float

    def to_json(self, filename: str):
        with open(filename, "w") as f:
            json.dump(asdict(self), f, indent=4)
