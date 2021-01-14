import logging
import re
import typing
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def train_and_evaluate(
    df,
    classifier,
    vectorizer,
    random_search=False,
    param_grid=None,
    cv=5,
    n_iter_search=10,
):

    model = Pipeline(steps=[("vectorizer", vectorizer), ("classifier", classifier)])
    le = LabelEncoder()
    df["target"] = le.fit_transform(df["sentiment"])

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"].values,
        df["target"].values.reshape(-1, 1),
        test_size=0.2,
        random_state=0,
    )

    if random_search and param_grid:
        model = RandomizedSearchCV(
            model, param_grid, cv=cv, n_iter=n_iter_search, n_jobs=-1, refit=True
        )

    model.fit(X_train, y_train)

    logger.info("model score: %.3f" % model.score(X_test, y_test))

    y_pred = model.predict(X_test)

    scores = metrics.classification_report(y_test, y_pred, output_dict=True)

    return model, le, scores


class SentimixModel:
    def __init__(
        self,
        vectorizer: typing.Any = None,
        classifier: typing.Any = None,
        model_path: str = None,
    ):
        if classifier is None and model_path is None:
            logger.exception("Please specify atleast one of classifer or model_path")
        self._model = None
        self.vectorizer = vectorizer
        self.classifier = classifier
        self._model_path = model_path
        if self._model_path is not None:
            self.load()

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        random_search: bool = False,
        param_grid: dict = {},
        cv: int = 5,
        n_iter_search: int = 10,
    ):
        model_pipeline = Pipeline(
            steps=[("vectorizer", self.vectorizer), ("classifier", self.classifier)]
        )

        if random_search and param_grid:
            self._model = RandomizedSearchCV(
                model_pipeline,
                param_grid,
                cv=cv,
                n_iter=n_iter_search,
                n_jobs=-1,
                refit=True,
            )
        else:
            self._model = model_pipeline

        self._model.fit(X, y)

        logger.info("training accuracy: %.3f" % self._model.score(X, y))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def evaluate(
        self, X: np.ndarray, y: np.ndarray, output_dict: bool = False
    ) -> typing.Union[dict, str]:
        y_pred = self._model.predict(X)
        scores = metrics.classification_report(y, y_pred, output_dict=output_dict)
        return scores

    def save(self, model_path: str = None):
        if model_path is not None:
            self._model_path = model_path

        if self._model is not None:
            if hasattr(self._model, "best_estimator_"):
                joblib.dump(self._model.best_estimator_, self._model_path)
            else:
                joblib.dump(self._model, self._model_path)

        else:
            logger.exception("The model is not trained yet, use .train() before saving")

    def load(self):
        try:
            self._model = joblib.load(self._model_path)
        except:
            logger.exception("error in loading the model")
            self._model = None
        return self
