"""Two-part lap-time model: race pace level from past seasons + within-race shape from a linear model."""
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from features import TyreTerms

TARGET = "DeltaToQuali"
SHAPE_COLS = ["Compound", "Circuit", "TyreAge", "LapsRemaining", "TrackTemp", "AirTemp"]


class TwoPartLapModel:
    """Predict DeltaToQuali (lap time minus the driver's best qualifying lap, in seconds).

    Part 1 - level: the median DeltaToQuali at this circuit in the training seasons
             (the all-circuit median for a circuit never seen before).
    Part 2 - shape: a Ridge model of how each lap differs from that driver's typical lap
             in the same race: compound pace, compound wear rates, optional circuit wear
             rates, fuel and temperature. It is trained on (lap - that driver's median lap
             in that race), so it learns only from same-driver, same-race comparisons.

    The driver's race median is used only to build the training target. Predictions use
    the circuit level from past seasons, never information from the race being predicted.
    """

    def __init__(self, alpha=1.0, circuit_slopes=True):
        self.alpha = alpha
        self.circuit_slopes = circuit_slopes

    def _make_shape_model(self):
        prep = ColumnTransformer([
            ("compound", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Compound"]),
            ("num", "passthrough", make_column_selector(dtype_include="number")),
        ])
        return Pipeline([
            ("terms", TyreTerms(compound_slopes=True, squares=False, circuit_slopes=self.circuit_slopes)),
            ("prep", prep),
            ("model", Ridge(alpha=self.alpha)),
        ])

    def fit(self, df):
        driver_race = ["Year", "Round", "Driver"]
        relative = df[TARGET] - df.groupby(driver_race)[TARGET].transform("median")
        self.circuit_level_ = df.groupby("Circuit")[TARGET].median()
        self.global_level_ = float(df[TARGET].median())
        self.shape_model_ = self._make_shape_model().fit(df[SHAPE_COLS], relative)
        return self

    def predict_level(self, X):
        return X["Circuit"].map(self.circuit_level_).fillna(self.global_level_).to_numpy(dtype=float)

    def predict_shape(self, X):
        return self.shape_model_.predict(X[SHAPE_COLS])

    def predict(self, X):
        return self.predict_level(X) + self.predict_shape(X)
