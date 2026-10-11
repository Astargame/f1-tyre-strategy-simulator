"""Feature helpers shared by the notebooks and the Streamlit app."""
from sklearn.base import BaseEstimator, TransformerMixin

DRY_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]


class TyreTerms(BaseEstimator, TransformerMixin):
    """Add tyre-wear columns so a linear model can learn how wear differs.

    - Age_<COMPOUND>:   tyre age on that compound (0 otherwise)    -> a wear rate per compound
    - AgeSq_<COMPOUND>: tyre age squared / 100 on that compound   -> wear that speeds up late in a stint
    - Age_<Circuit>:    tyre age at that circuit (0 otherwise)     -> a wear rate per track

    The list of circuits is learned in fit() from the training data only,
    so an unseen circuit simply gets no circuit-specific wear column.
    """

    def __init__(self, compound_slopes=True, squares=True, circuit_slopes=True):
        self.compound_slopes = compound_slopes
        self.squares = squares
        self.circuit_slopes = circuit_slopes

    def fit(self, X, y=None):
        self.circuits_ = sorted(X["Circuit"].unique())
        return self

    def transform(self, X):
        X = X.copy()
        age = X["TyreAge"].astype(float)
        for comp in DRY_COMPOUNDS:
            is_comp = (X["Compound"] == comp).astype(float)
            if self.compound_slopes:
                X[f"Age_{comp}"] = age * is_comp
            if self.squares:
                X[f"AgeSq_{comp}"] = (age ** 2) * is_comp / 100
        if self.circuit_slopes:
            for circuit in self.circuits_:
                X[f"Age_{circuit}"] = age * (X["Circuit"] == circuit).astype(float)
        return X
