import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Union
from scipy.optimize import least_squares

from .models.base_models import HierarchicalDSRModel


def _power_law(x: np.ndarray, a: float, b: float, c: float, p: float, side: str) -> np.ndarray:
    if side == "left":
        dx = x - b
    elif side == "right":
        dx = b - x
    else:
        raise ValueError(f"Unsupported side: {side}")
    return a * np.power(np.clip(dx, 0.0, None), p) + c


def _signed_power_law(x: np.ndarray, a: float, b: float, c: float, p: float) -> np.ndarray:
    dx = x - b
    return a * np.sign(dx) * np.power(np.abs(dx), p) + c


def _inlier_mask(residuals: np.ndarray, outlier_k: Optional[float], max_drop_frac: float) -> np.ndarray:
    mask = np.ones(residuals.size, dtype=bool)
    max_drop = min(int(np.ceil(max_drop_frac * residuals.size)), residuals.size - 4)
    if outlier_k is None or max_drop <= 0:
        return mask

    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    if rmse < 1e-12:
        return mask

    scaled = np.abs(residuals) / rmse
    outliers = scaled > outlier_k
    if outliers.any():
        mask[np.argsort(scaled)[-min(int(outliers.sum()), max_drop):]] = False
    return mask


def _solve_power_law(
    value_fn,
    x: np.ndarray,
    y: np.ndarray,
    seed: Tuple[float, ...],
    p_bounds: Tuple[float, float],
    b_bounds: Tuple[float, float] = (-np.inf, np.inf),
) -> dict:
    lower = np.array([-np.inf, b_bounds[0], -np.inf, p_bounds[0]], dtype=float)
    upper = np.array([np.inf, b_bounds[1], np.inf, p_bounds[1]], dtype=float)

    def residuals(theta: np.ndarray) -> np.ndarray:
        return value_fn(x, *theta) - y

    try:
        result = least_squares(
            residuals,
            x0=np.array(seed, dtype=float),
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=max(float(np.median(np.abs(y - np.median(y)))) * 1.4826, 1e-6),
            max_nfev=5000,
        )
        params = tuple(float(v) for v in result.x)
        residual = residuals(result.x)
    except (RuntimeError, ValueError, FloatingPointError):
        params = tuple(float(v) for v in seed)
        residual = residuals(np.array(params, dtype=float))

    return {"params": params, "score": float(np.mean(np.abs(residual)))}


def _fit_power_law(
    x: np.ndarray,
    y: np.ndarray,
    p_bounds: Tuple[float, float],
    outlier_k: Optional[float],
    max_drop_frac: float,
    extrapolation_bounds: Optional[Tuple[float, float]] = None,
) -> dict:
    p_min, p_max = p_bounds
    lower_bound = float(np.min(x))
    upper_bound = float(np.max(x))
    if extrapolation_bounds is not None:
        lower_bound = min(lower_bound, float(extrapolation_bounds[0]))
        upper_bound = max(upper_bound, float(extrapolation_bounds[1]))

    margin = max(1e-3 * float(upper_bound - lower_bound), 1e-3)
    candidates = []

    for side, b in (("left", lower_bound - margin), ("right", upper_bound + margin)):
        x0 = x - b if side == "left" else b - x
        a, c = np.polyfit(x0, y, 1)
        def value_fn(x_eval, a, b, c, p, side=side):
            return _power_law(x_eval, a, b, c, p, side)
        b_bounds = (-np.inf, b) if side == "left" else (b, np.inf)
        for p in (1.0, p_min, p_max, 0.5 * (p_min + p_max)):
            fit = _solve_power_law(value_fn, x, y, (float(a), b, float(c), p), p_bounds, b_bounds)
            fit["side"] = side
            candidates.append(fit)

    best = min(candidates, key=lambda fit: fit["score"])
    a, b, c, p = best["params"]
    mask = _inlier_mask(y - _power_law(x, a, b, c, p, best["side"]), outlier_k, max_drop_frac)
    if not mask.all():
        best = _fit_power_law(x[mask], y[mask], p_bounds, None, 0.0, extrapolation_bounds)
    best["inlier_mask"] = mask
    return best


def _fit_signed_power_law(
    x: np.ndarray,
    y: np.ndarray,
    p_bounds: Tuple[float, float],
    outlier_k: Optional[float],
    max_drop_frac: float,
) -> dict:
    p_min, p_max = p_bounds
    candidates = []
    b_values = np.linspace(float(np.min(x)), float(np.max(x)), num=min(max(x.size, 3), 7))

    for b in b_values:
        a = float(np.polyfit(x - b, y, 1)[0])
        c = float(np.mean(y))
        for p in (1.0, p_min, p_max, 0.5 * (p_min + p_max)):
            candidates.append(_solve_power_law(_signed_power_law, x, y, (a, float(b), c, p), p_bounds))

    best = min(candidates, key=lambda fit: fit["score"])
    mask = _inlier_mask(y - _signed_power_law(x, *best["params"]), outlier_k, max_drop_frac)
    if not mask.all():
        best = _fit_signed_power_law(x[mask], y[mask], p_bounds, None, 0.0)
    best["inlier_mask"] = mask
    return best


def _fit_family(
    family: str,
    x: np.ndarray,
    y: np.ndarray,
    p_bounds: Tuple[float, float],
    outlier_k: Optional[float],
    max_drop_frac: float,
    extrapolation_bounds: Optional[Tuple[float, float]] = None,
) -> dict:
    if family == "polynomial":
        degree = max(1, int(round(p_bounds[1])))
        coeffs = tuple(float(v) for v in np.polyfit(x, y, deg=degree))
        return {"family": family, "params": coeffs, "degree": degree}
    if family == "power law":
        fit = _fit_power_law(x, y, p_bounds, outlier_k, max_drop_frac, extrapolation_bounds)
    elif family == "signed power law":
        fit = _fit_signed_power_law(x, y, p_bounds, outlier_k, max_drop_frac)
    else:
        raise ValueError(f"Unsupported function family: {family}")
    fit["family"] = family
    return fit


def _predict_family(fit: dict, x: np.ndarray) -> np.ndarray:
    if fit["family"] == "polynomial":
        return np.polyval(np.asarray(fit["params"], dtype=float), x)
    if fit["family"] == "power law":
        a, b, c, p = fit["params"]
        return _power_law(x, a, b, c, p, fit["side"])
    if fit["family"] == "signed power law":
        return _signed_power_law(x, *fit["params"])
    raise ValueError(f"Unsupported function family: {fit['family']}")


class FeatureExtrapolation(nn.Module):
    def __init__(self, model: HierarchicalDSRModel) -> None:
        super().__init__()
        self.model = model
        self.feature_splitting = model.feature_splitting
        self.feature_dim = model.features_dyn.shape[1] if self.feature_splitting else model.features.shape[1]

        self._fit_specs = {}
        self._fit_specs_pos = {}
        self.fitted_function_family = None
        self.fitted = False
        self.coefficients = {}

    def _fit_feature_block(
        self,
        x: np.ndarray,
        features: np.ndarray,
        families: list[str],
        p_bounds: Tuple[float, float],
        outlier_k: Optional[float],
        max_drop_frac: float,
        extrapolation_bounds: Optional[Tuple[float, float]],
        n_splits: int,
    ) -> Tuple[dict, dict, dict, dict]:
        from sklearn.model_selection import KFold

        best_families = {}
        cv_scores = {}
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42) if n_splits >= 2 else None

        for feature_idx in range(features.shape[1]):
            if kfold is None:
                best_families[feature_idx] = families[0]
                cv_scores[feature_idx] = float("nan")
                continue

            scores = {}
            for family in families:
                errors = []
                for train_idx, val_idx in kfold.split(x):
                    fit = _fit_family(
                        family,
                        x[train_idx],
                        features[train_idx, feature_idx],
                        p_bounds,
                        outlier_k,
                        max_drop_frac,
                        extrapolation_bounds,
                    )
                    pred = _predict_family(fit, x[val_idx])
                    errors.append(float(np.mean(np.abs(pred - features[val_idx, feature_idx]))))
                scores[family] = float(np.mean(errors))

            best_families[feature_idx] = min(scores, key=lambda family: scores[family])
            cv_scores[feature_idx] = scores[best_families[feature_idx]]

        fit_specs = {}
        coefficients = {}
        for feature_idx, family in best_families.items():
            fit = _fit_family(
                family,
                x,
                features[:, feature_idx],
                p_bounds,
                outlier_k,
                max_drop_frac,
                extrapolation_bounds,
            )
            fit_specs[feature_idx] = fit
            coefficients[feature_idx] = fit["params"]

        return best_families, cv_scores, coefficients, fit_specs

    def _prepare_control(self, control_parameters: torch.Tensor) -> np.ndarray:
        x = control_parameters.detach().cpu().numpy().reshape(control_parameters.shape[0], -1)
        if x.shape[1] != 1:
            raise ValueError("FeatureExtrapolation currently supports exactly one control parameter.")
        return x[:, 0].astype(float, copy=False)

    def _predict_feature_block(
        self,
        x: np.ndarray,
        fit_specs: dict,
        template: torch.Tensor,
    ) -> torch.Tensor:
        features = np.zeros((x.shape[0], len(fit_specs)), dtype=float)
        for feature_idx, fit in fit_specs.items():
            features[:, feature_idx] = _predict_family(fit, x)
        return torch.as_tensor(features, dtype=template.dtype, device=template.device)

    def extrapolate(self, control_parameters: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Predict feature values for new control parameters.

        control_parameters: (num_subjects, num_control_parameters)
        returns: features, or (features_dyn, features_pos) if feature_splitting is True
        """
        if not self.fitted:
            raise RuntimeError("Call fit before extrapolating feature values.")

        x = self._prepare_control(control_parameters)
        if self.feature_splitting:
            features_dyn = self._predict_feature_block(x, self._fit_specs, self.model.features_dyn)
            features_pos = self._predict_feature_block(x, self._fit_specs_pos, self.model.features_pos)
            return features_dyn, features_pos
        return self._predict_feature_block(x, self._fit_specs, self.model.features)

    def predict(self, control_parameters: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self.extrapolate(control_parameters)

    def fit(
        self,
        control_parameters: torch.Tensor,
        extrapolation_control_parameters: Optional[torch.Tensor] = None,
        function_families: list[str] = ["polynomial", "power law", "signed power law"],
        max_degree: float = 2.0,
        min_degree: float = 0.25,
        outlier_k: Optional[float] = 2.5,
        max_drop_frac: float = 0.05,
    ) -> dict:
        """
        Fit one feature/control-parameter relation per learned feature.

        control_parameters: (num_subjects, num_control_parameters)
        extrapolation_control_parameters: optional OOD control parameters used to constrain power-law critical values outside the combined ID/OOD interval
        function_families: list of function families in ["polynomial", "poly", "power law", "pl", "signed power law", "spl"]
        max_degree: upper exponent bound for power-law fits and polynomial degree
        min_degree: lower exponent bound for power-law fits
        """
        aliases = {
            "polynomial": "polynomial",
            "poly": "polynomial",
            "power law": "power law",
            "pl": "power law",
            "signed power law": "signed power law",
            "spl": "signed power law",
        }
        families = []
        for family in function_families:
            if family not in aliases:
                raise ValueError(f"Unsupported function family: {family}")
            families.append(aliases[family])

        x = self._prepare_control(control_parameters)
        extrapolation_bounds = None
        if extrapolation_control_parameters is not None:
            x_extrap = self._prepare_control(extrapolation_control_parameters)
            extrapolation_bounds = (float(np.min(x_extrap)), float(np.max(x_extrap)))

        args = (
            families,
            (float(min_degree), float(max_degree)),
            outlier_k,
            max_drop_frac,
            extrapolation_bounds,
            min(5, x.shape[0]),
        )
        if self.feature_splitting:
            dyn = self._fit_feature_block(x, self.model.features_dyn.detach().cpu().numpy(), *args)
            pos = self._fit_feature_block(x, self.model.features_pos.detach().cpu().numpy(), *args)
            dyn_families, dyn_scores, dyn_coeffs, dyn_specs = dyn
            pos_families, pos_scores, pos_coeffs, pos_specs = pos

            self._fit_specs = dyn_specs
            self._fit_specs_pos = pos_specs
            self.fitted_function_family = {"dynamic": dyn_families, "positional": pos_families}
            self.coefficients = {"dynamic": dyn_coeffs, "positional": pos_coeffs}
            cv_score = {"dynamic": dyn_scores, "positional": pos_scores}
            fit_specs = {"dynamic": dyn_specs, "positional": pos_specs}
        else:
            best_families, cv_score, coefficients, fit_specs = self._fit_feature_block(
                x,
                self.model.features.detach().cpu().numpy(),
                *args,
            )
            self._fit_specs = fit_specs
            self._fit_specs_pos = {}
            self.fitted_function_family = best_families
            self.coefficients = coefficients

        self.fitted = True
        return {
            "function_family": self.fitted_function_family,
            "coefficients": self.coefficients,
            "cv_score": cv_score,
            "fit_specs": fit_specs,
        }
    
    