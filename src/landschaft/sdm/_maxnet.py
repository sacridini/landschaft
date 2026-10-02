# SPDX-License-Identifier: GPL-3.0-or-later
"""MaxNet: Maxent as R's maxnet fits it (glmnet's lasso path), as a scikit-learn estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import _core
from .._threads import limit

__all__ = ["MaxNet"]

try:  # scikit-learn is optional: inherit its mixins when it is there
    from sklearn.base import BaseEstimator as _Base
    from sklearn.base import ClassifierMixin as _Classifier
except ImportError:  # pragma: no cover - exercised without scikit-learn
    _Base = object

    class _Classifier:
        _estimator_type = "classifier"

_TYPES = ("link", "exponential", "cloglog", "logistic")
_KIND_LINEAR, _KIND_QUADRATIC, _KIND_HINGE, _KIND_THRESHOLD, _KIND_CATEGORICAL, _KIND_PRODUCT = range(6)


def _label(value) -> str:
    """A factor level as R writes it (as.character)."""
    if isinstance(value, (bool, np.bool_)):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return _core.sdm_r_as_character(float(value))
    return str(value)


class MaxNet(_Classifier, _Base):
    """Maxent fitted as R's maxnet does it, natively: the same model, faster.

    Maxent (Phillips et al. 2006) as an infinitely weighted logistic
    regression (Phillips et al. 2017): presences against background points
    with the features of maxnet (linear ``l``, quadratic ``q``, product
    ``p``, hinge ``h``, threshold ``t``, categorical), maxnet's default
    regularisation and glmnet's lasso path, keeping its last lambda. Every
    step is a C++ port of R's maxnet 0.1.4 and glmnet 4.1-10 that gives R's
    results bit for bit on x86-64 (coefficients, alpha, entropy, the four
    output types with and without clamping), many times faster (hashing of
    the duplicate rows, the gradient checks screened by a bound and run in
    parallel, fused residual updates).

    A scikit-learn classifier (``fit``, ``predict_proba``, ``predict``,
    ``decision_function``, ``get_params`` / ``set_params``, cloning and
    pickling), usable in ``GridSearchCV`` and pipelines; scikit-learn itself
    is not needed.

    Parameters
    ----------
    classes : str
        Feature classes: ``"default"`` (maxnet's rule by the number of
        presences: ``"l"`` below 10, ``"lq"`` below 15, ``"lqh"`` below 80,
        else ``"lqph"``) or letters among ``l``, ``q``, ``p``, ``h``, ``t``.
        Categorical predictors always give one indicator per level.
    regmult : float
        Regularisation multiplier (larger: smoother, fewer features).
    add_samples_to_background : bool
        Add the presences to the background, unless the same values already
        are there (maxnet's ``addsamplestobackground``).
    type : {"cloglog", "logistic", "exponential", "link"}
        Output of :meth:`predict_proba` (column 1) and :meth:`predict`:
        cloglog (default) and logistic are suitabilities in [0, 1].
    clamp : bool
        Clamp predictors and features to their training ranges when
        predicting.
    categorical : list of str or int, optional
        Columns that are categorical (besides those of pandas ``category``
        or text dtype). Their levels are those of the category dtype, else
        the sorted distinct values.
    threshold : float
        Suitability at or above which :meth:`predict` gives 1 (e.g. one of
        :func:`ls.sdm.threshold`'s).
    n_threads : int, optional
        Threads (None or 0: all cores but one). The result does not depend
        on it.

    Attributes
    ----------
    betas_ : dict
        Feature name (maxnet's column names, e.g. ``"hinge(bio1):12.5:28"``)
        -> coefficient, for the features with a nonzero coefficient.
    alpha_ : float
        Normalising constant (the exponential output sums to 1 over the
        background).
    entropy_ : float
        Entropy of the exponential output over the background.
    lambda_ : float
        The lambda of the model (the last of glmnet's path).
    lambdas_ : numpy.ndarray
        The 200 lambdas of the path.
    penalty_factor_ : numpy.ndarray
        Regularisation of every feature (``feature_names_``).
    feature_names_ : list of str
        Every feature (column of maxnet's model matrix).
    feature_classes_ : str
        The feature classes used (``"default"`` resolved).
    feature_names_in_ : numpy.ndarray
        The predictor names (when fitted on a DataFrame).
    n_features_in_ : int
    classes_ : numpy.ndarray
        ``[0, 1]``.
    n_passes_ : int
        Coordinate-descent passes of the path (glmnet's ``npasses``).
    var_min_, var_max_ : dict
        Range of each continuous predictor (clamping).
    sample_means_ : dict
        Mean of each continuous predictor and most frequent level of each
        categorical one over the presences (maxnet's ``samplemeans``).
    levels_ : dict
        Levels of each categorical predictor.

    Notes
    -----
    ``sample_weight`` is accepted for compatibility and ignored: maxnet
    weighs presences 1 and background points 100. Rows with a missing value
    are not allowed in ``fit``; ``predict_*`` gives NaN for them (R drops
    them). Duplicate presences among the background are found by exact
    comparison of the values (R compares formatted text when there are
    categorical predictors, which can miss some).

    References
    ----------
    Phillips, S. J., Anderson, R. P., Dudík, M., Schapire, R. E. & Blair,
    M. E. (2017). Opening the black box: an open-source release of Maxent.
    Ecography 40, 887-893.

    Friedman, J., Hastie, T. & Tibshirani, R. (2010). Regularization paths
    for generalized linear models via coordinate descent. Journal of
    Statistical Software 33, 1-22.

    Examples
    --------
    >>> import landschaft as ls
    >>> env = ls.load_raster(ls.sdm.example_data("bioclim"))
    >>> occ = ls.load_table(ls.sdm.example_data("bradypus"))
    >>> bg = ls.sdm.background(env, 5000, occ=occ, seed=1)
    >>> data = ls.sdm.dataset(occ, bg, env)
    >>> model = ls.sdm.MaxNet().fit(data.X, data.y)
    >>> model.feature_classes_, len(model.betas_)
    ('lqph', 29)
    >>> suitability = model.predict_proba(data.X)[:, 1]   # cloglog, in [0, 1]
    >>> bool(suitability[data.y == 1].mean() > suitability[data.y == 0].mean())
    True
    """

    def __init__(self, classes: str = "default", regmult: float = 1.0, add_samples_to_background: bool = True,
                 type: str = "cloglog", clamp: bool = True, categorical=None, threshold: float = 0.5,
                 n_threads: int | None = None):
        self.classes = classes
        self.regmult = regmult
        self.add_samples_to_background = add_samples_to_background
        self.type = type
        self.clamp = clamp
        self.categorical = categorical
        self.threshold = threshold
        self.n_threads = n_threads

    # -- scikit-learn's parameter protocol, when scikit-learn is not there --------------------------

    if _Base is object:  # pragma: no cover - exercised without scikit-learn
        _PARAMS = ("classes", "regmult", "add_samples_to_background", "type", "clamp", "categorical", "threshold",
                   "n_threads")

        def get_params(self, deep: bool = True) -> dict:
            return {k: getattr(self, k) for k in self._PARAMS}

        def set_params(self, **params):
            for k, v in params.items():
                if k not in self._PARAMS:
                    raise ValueError(f"invalid parameter {k!r} for MaxNet")
                setattr(self, k, v)
            return self

        def __repr__(self) -> str:
            default = MaxNet()
            args = [f"{k}={getattr(self, k)!r}" for k in self._PARAMS if getattr(self, k) != getattr(default, k)]
            return f"MaxNet({', '.join(args)})"

    # -- the predictors as two tables (continuous, categorical) -------------------------------------

    def _columns(self, X):
        if isinstance(X, pd.DataFrame):
            return X, [str(c) for c in X.columns], True
        a = np.asarray(X)
        if a.ndim == 1:
            a = a.reshape(-1, 1)
        if a.ndim != 2:
            raise ValueError("X must be a 2-D table (rows x predictors)")
        frame = pd.DataFrame(a, columns=[f"x{j}" for j in range(a.shape[1])])
        return frame, list(frame.columns), False

    def _is_categorical(self, frame: pd.DataFrame, j: int) -> bool:
        s = frame.iloc[:, j]
        listed = self.categorical or []
        if isinstance(listed, (str, int)):
            listed = [listed]
        if str(frame.columns[j]) in [str(c) for c in listed] or j in [c for c in listed if isinstance(c, int)]:
            return True
        return isinstance(s.dtype, pd.CategoricalDtype) or not (
            pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s))

    @staticmethod
    def _levels(s: pd.Series) -> list:
        if isinstance(s.dtype, pd.CategoricalDtype):
            return list(s.dtype.categories)
        values = s.dropna().unique()
        try:
            return sorted(values)
        except TypeError:
            return sorted(values, key=str)

    @staticmethod
    def _codes(s: pd.Series, levels: list) -> np.ndarray:
        values = s.astype(object) if isinstance(s.dtype, pd.CategoricalDtype) else s
        codes = pd.Index(levels).get_indexer(values).astype(np.int32)
        codes[(codes == -1) & s.notna().to_numpy()] = -2  # a level not seen in training
        return codes

    def _tables(self, frame: pd.DataFrame, cont_idx, cat_idx, levels):
        n = len(frame)
        cont = np.empty((n, len(cont_idx)), dtype=np.float64, order="F")
        for k, j in enumerate(cont_idx):
            cont[:, k] = pd.to_numeric(frame.iloc[:, j], errors="coerce").to_numpy(dtype=np.float64,
                                                                                    na_value=np.nan)
        cat = np.empty((n, len(cat_idx)), dtype=np.int32, order="F")
        for k, j in enumerate(cat_idx):
            cat[:, k] = self._codes(frame.iloc[:, j], levels[k])
        return cont, cat

    # -- fitting ------------------------------------------------------------------------------------

    def fit(self, X, y=None, sample_weight=None):
        """Fit to presences (``y`` 1) and background points (``y`` 0).

        Parameters
        ----------
        X : pandas.DataFrame or array-like (n, p)
            Predictors; an :class:`ls.sdm.SDMData` (then ``y`` is its ``y``).
        y : array-like (n,)
            1 for a presence, 0 for a background point.
        sample_weight : ignored
            maxnet's weights are fixed (presences 1, background 100).

        Returns
        -------
        MaxNet
            self
        """
        if y is None and hasattr(X, "X") and hasattr(X, "y"):
            X, y = X.X, X.y
        if y is None:
            raise ValueError("fit needs y (1 presence, 0 background)")
        if self.type not in _TYPES:
            raise ValueError(f"type must be one of {', '.join(_TYPES)}")
        frame, names, is_frame = self._columns(X)
        p = np.asarray(y, dtype=np.float64).ravel()
        if p.shape != (len(frame),):
            raise ValueError(f"y must have one value per row of X ({len(frame)})")
        if not np.isin(p, (0.0, 1.0)).all():
            raise ValueError("y must be 1 (presence) or 0 (background)")
        cat_idx = [j for j in range(frame.shape[1]) if self._is_categorical(frame, j)]
        cont_idx = [j for j in range(frame.shape[1]) if j not in cat_idx]
        levels = [self._levels(frame.iloc[:, j]) for j in cat_idx]
        cont, cat = self._tables(frame, cont_idx, cat_idx, levels)
        if np.isnan(cont).any() or (cat < 0).any():
            raise ValueError("MaxNet cannot fit rows with missing values: remove them first")
        with limit(self.n_threads):
            m = _core.sdm_maxnet_fit(cont, cat, [len(lv) for lv in levels], p, str(self.classes),
                                     float(self.regmult), bool(self.add_samples_to_background))
        cont_names = [names[j] for j in cont_idx]
        cat_names = [names[j] for j in cat_idx]
        labels = [[_label(v) for v in lv] for lv in levels]
        self.feature_classes_ = m["classes"]
        self.feature_names_ = self._feature_names(m, cont_names, cat_names, labels)
        bi = np.asarray(m["beta_index"], dtype=np.int64)
        self._model_ = {
            "kind": m["kind"][bi], "var": m["var"][bi], "var2": m["var2"][bi], "pk1": m["pk1"][bi],
            "pk2": m["pk2"][bi], "betas": m["betas"], "feature_min": m["feature_min"][bi],
            "feature_max": m["feature_max"][bi], "var_min": m["var_min"], "var_max": m["var_max"],
            "alpha": float(m["alpha"]), "entropy": float(m["entropy"]), "cont_idx": cont_idx, "cat_idx": cat_idx,
            "levels": levels,
        }
        self.betas_ = {self.feature_names_[j]: float(b) for j, b in zip(bi, m["betas"])}
        self.alpha_ = float(m["alpha"])
        self.entropy_ = float(m["entropy"])
        self.lambdas_ = np.asarray(m["lambda"])
        self.lambda_ = float(self.lambdas_[-1])
        self.penalty_factor_ = np.asarray(m["penalty"])
        self.feature_mins_ = np.asarray(m["feature_min"])
        self.feature_maxs_ = np.asarray(m["feature_max"])
        self.n_passes_ = int(m["n_passes"])
        self.n_added_ = int(m["n_added"])
        self.var_min_ = dict(zip(cont_names, map(float, m["var_min"])))
        self.var_max_ = dict(zip(cont_names, map(float, m["var_max"])))
        means = dict(zip(cont_names, map(float, m["sample_means"])))
        for name, lv, k in zip(cat_names, levels, m["majority"]):
            means[name] = lv[int(k)]
        self.sample_means_ = means
        self.levels_ = dict(zip(cat_names, levels))
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = frame.shape[1]
        if is_frame:
            self.feature_names_in_ = np.asarray(names, dtype=object)
        elif hasattr(self, "feature_names_in_"):
            del self.feature_names_in_
        return self

    @staticmethod
    def _feature_names(m: dict, cont: list, cat: list, labels: list) -> list:
        names = []
        plain = "l" in m["classes"]
        for kind, a, b, s1, s2 in zip(m["kind"], m["var"], m["var2"], m["s1"], m["s2"]):
            if kind == _KIND_LINEAR:
                names.append(cont[a])
            elif kind == _KIND_QUADRATIC:
                names.append(f"I({cont[a]}^2)")
            elif kind == _KIND_HINGE:
                names.append(f"hinge({cont[a]}):{s1}:{s2}")
            elif kind == _KIND_THRESHOLD:
                names.append(f"thresholds({cont[a]}):{s1}")
            elif kind == _KIND_CATEGORICAL:
                names.append(f"categorical({cat[a]}):{labels[a][b]}")
            else:
                # model.matrix names a product by the order of the formula's
                # variables: without linear terms the first product term
                # (cont[1]:cont[0]) puts cont[1] first
                first, second = (b, a) if (not plain and (a, b) == (0, 1)) else (a, b)
                names.append(f"{cont[first]}:{cont[second]}")
        return names

    # -- predicting ---------------------------------------------------------------------------------

    def _check_fitted(self):
        if not hasattr(self, "_model_"):
            raise AttributeError("this MaxNet is not fitted yet: call fit() first")

    def _frame(self, X) -> pd.DataFrame:
        if hasattr(X, "X") and hasattr(X, "y") and not isinstance(X, (pd.DataFrame, np.ndarray)):
            X = X.X
        if isinstance(X, pd.DataFrame) and hasattr(self, "feature_names_in_"):
            missing = [c for c in self.feature_names_in_ if c not in [str(k) for k in X.columns]]
            if missing:
                raise ValueError(f"X lacks the predictors {missing}")
            X = X.rename(columns=str)
            return X[list(self.feature_names_in_)]
        frame, _, _ = self._columns(X)
        if frame.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {frame.shape[1]} columns, the model {self.n_features_in_}")
        return frame

    def predict_output(self, X, type: str | None = None, clamp: bool | None = None) -> np.ndarray:
        """maxnet's predict(): the output of one type for every row.

        Parameters
        ----------
        X : pandas.DataFrame or array-like (n, p)
            Predictors (the columns of ``fit``, by name for a DataFrame).
        type : {"cloglog", "logistic", "exponential", "link"}, optional
            Default: the estimator's ``type``.
        clamp : bool, optional
            Default: the estimator's ``clamp``.

        Returns
        -------
        numpy.ndarray (n,)
            NaN where a predictor the model uses is missing.
        """
        self._check_fitted()
        type = self.type if type is None else type
        if type not in _TYPES:
            raise ValueError(f"type must be one of {', '.join(_TYPES)}")
        clamp = self.clamp if clamp is None else clamp
        frame = self._frame(X)
        md = self._model_
        cont, cat = self._tables(frame, md["cont_idx"], md["cat_idx"], md["levels"])
        with limit(self.n_threads):
            return _core.sdm_maxnet_predict(md["kind"], md["var"], md["var2"], md["pk1"], md["pk2"], md["betas"],
                                            md["feature_min"], md["feature_max"], md["var_min"], md["var_max"],
                                            md["alpha"], md["entropy"], cont, cat,
                                            [len(lv) for lv in md["levels"]], bool(clamp), type)

    def predict_proba(self, X) -> np.ndarray:
        """Suitability as a probability: column 1 the ``type`` output (cloglog or logistic), column 0 its complement.

        Returns
        -------
        numpy.ndarray (n, 2)
        """
        if self.type not in ("cloglog", "logistic"):
            raise ValueError("predict_proba needs type 'cloglog' or 'logistic' (see predict_output for the others)")
        s = self.predict_output(X)
        return np.column_stack([1.0 - s, s])

    def predict(self, X) -> np.ndarray:
        """1 where the ``type`` output reaches ``threshold``, else 0 (NaN rows: 0).

        Returns
        -------
        numpy.ndarray (n,) of int
        """
        s = self.predict_output(X)
        with np.errstate(invalid="ignore"):
            return (s >= self.threshold).astype(np.int64)

    def decision_function(self, X) -> np.ndarray:
        """The link (linear predictor plus alpha), clamped as ``clamp`` says.

        Returns
        -------
        numpy.ndarray (n,)
        """
        return self.predict_output(X, type="link")

    def __sklearn_tags__(self):  # pragma: no cover - scikit-learn >= 1.6
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.input_tags.categorical = True
        tags.input_tags.string = True
        return tags
