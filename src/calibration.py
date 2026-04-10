"""
Calibration post-hoc pour modeles surentraines sur donnees SMOTE.

La calibration doit etre apprise sur des donnees reelles (non sur-echantillonnees)
pour que les probabilites refletent la vraie distribution des classes.
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class IsotonicCalibratedModel:
    """
    Enveloppe appliquant une calibration par regression isotonique
    apprise sur un jeu de calibration aux vraies proportions de classes.
    """

    def __init__(self, base_model, iso: IsotonicRegression):
        self.base_model = base_model
        self.iso = iso

    def predict_proba(self, X):
        raw = self.base_model.predict_proba(X)[:, 1]
        cal = self.iso.predict(raw)
        cal = np.clip(cal, 0.0, 1.0)
        return np.column_stack([1 - cal, cal])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @classmethod
    def fit(cls, base_model, X_cal, y_cal):
        """
        Apprend la calibration isotonique sur X_cal, y_cal
        (donnees reelles, jamais vues par base_model pendant l'entrainement).
        """
        raw = base_model.predict_proba(X_cal)[:, 1]
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(raw, y_cal)
        return cls(base_model, iso)


class PlattCalibratedModel:
    """
    Enveloppe appliquant une calibration sigmoid (Platt scaling).
    Plus stable que la regression isotonique quand les probabilites brutes sont bimodales.
    """

    def __init__(self, base_model, platt: LogisticRegression):
        self.base_model = base_model
        self.platt = platt

    def predict_proba(self, X):
        raw = self.base_model.predict_proba(X)[:, 1]
        cal = self.platt.predict_proba(raw.reshape(-1, 1))[:, 1]
        cal = np.clip(cal, 0.0, 1.0)
        return np.column_stack([1 - cal, cal])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @classmethod
    def fit(cls, base_model, X_cal, y_cal):
        """
        Apprend la calibration sigmoid (Platt) sur X_cal, y_cal
        (donnees reelles, jamais vues par base_model pendant l'entrainement).
        """
        raw = base_model.predict_proba(X_cal)[:, 1]
        platt = LogisticRegression(C=1.0)
        platt.fit(raw.reshape(-1, 1), y_cal)
        return cls(base_model, platt)
