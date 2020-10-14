# Class Krige(RegressorMixin, BaseEstimator) 
#
# This class is an implementation of the PyKrige class that 
# supports optimization of the vparams parameters. It was provided to us
# By the PyKrige developers, in particular Sebastian Müller. (April, 2020)
# This code should become part of PyKrige 1.5.1
#

import numpy as np
from pykrige.compat import GridSearchCV
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
from pykrige.ok3d import OrdinaryKriging3D
from pykrige.uk3d import UniversalKriging3D
from sklearn.base import RegressorMixin, BaseEstimator

from utilities.utilities import utilities

krige_methods = {
    "ordinary": OrdinaryKriging,
    "universal": UniversalKriging,
    "ordinary3d": OrdinaryKriging3D,
    "universal3d": UniversalKriging3D,
}
threed_krige = ("ordinary3d", "universal3d")
# valid additional keywords for each method
krige_methods_kws = {
    "ordinary": [
        "anisotropy_scaling",
        "anisotropy_angle",
        "enable_statistics",
        "coordinates_type",
    ],
    "universal": [
        "anisotropy_scaling",
        "anisotropy_angle",
        "drift_terms",
        "point_drift",
        "external_drift",
        "external_drift_x",
        "external_drift_y",
        "functional_drift",
    ],
    "ordinary3d": [
        "anisotropy_scaling_y",
        "anisotropy_scaling_z",
        "anisotropy_angle_x",
        "anisotropy_angle_y",
        "anisotropy_angle_z",
    ],
    "universal3d": [
        "anisotropy_scaling_y",
        "anisotropy_scaling_z",
        "anisotropy_angle_x",
        "anisotropy_angle_y",
        "anisotropy_angle_z",
        "drift_terms",
        "functional_drift",
    ],
}

def validate_method(method):
    if method not in krige_methods.keys():
        raise ValueError(
            "Kriging method must be one of {}".format(krige_methods.keys())
        )

class Krige(RegressorMixin, BaseEstimator):
    def __init__(
        self,
        method="ordinary",
        variogram_model="linear",
        variogram_parameters=None,
        variogram_function=None,
        nlags=6,
        weight=False,
        n_closest_points=10,
        verbose=False,
        anisotropy_scaling=1.0,
        anisotropy_angle=0.0,
        enable_statistics=False,
        coordinates_type="euclidean",
        anisotropy_scaling_y=1.0,
        anisotropy_scaling_z=1.0,
        anisotropy_angle_x=0.0,
        anisotropy_angle_y=0.0,
        anisotropy_angle_z=0.0,
        drift_terms=None,
        point_drift=None,
        external_drift=None,
        external_drift_x=None,
        external_drift_y=None,
        functional_drift=None,
    ):
        validate_method(method)
        self.variogram_model = variogram_model
        self.variogram_parameters = variogram_parameters
        self.variogram_function = variogram_function
        self.nlags = nlags
        self.weight = weight
        self.verbose = verbose
        self.anisotropy_scaling = anisotropy_scaling
        self.anisotropy_angle = anisotropy_angle
        self.enable_statistics = enable_statistics
        self.coordinates_type = coordinates_type
        self.anisotropy_scaling_y = anisotropy_scaling_y
        self.anisotropy_scaling_z = anisotropy_scaling_z
        self.anisotropy_angle_x = anisotropy_angle_x
        self.anisotropy_angle_y = anisotropy_angle_y
        self.anisotropy_angle_z = anisotropy_angle_z
        self.drift_terms = drift_terms
        self.point_drift = point_drift
        self.external_drift = external_drift
        self.external_drift_x = external_drift_x
        self.external_drift_y = external_drift_y
        self.functional_drift = functional_drift
        self.model = None  # not trained
        self.n_closest_points = n_closest_points
        self.method = method
        self.val_kw = "val" if self.method in threed_krige else "z"

    def fit(self, x, y, *args, **kwargs):
        setup = dict(
            variogram_model=self.variogram_model,
            variogram_parameters=self.variogram_parameters,
            variogram_function=self.variogram_function,
            nlags=self.nlags,
            weight=self.weight,
            verbose=self.verbose,
        )
        add_setup = dict(
            anisotropy_scaling=self.anisotropy_scaling,
            anisotropy_angle=self.anisotropy_angle,
            enable_statistics=self.enable_statistics,
            coordinates_type=self.coordinates_type,
            anisotropy_scaling_y=self.anisotropy_scaling_y,
            anisotropy_scaling_z=self.anisotropy_scaling_z,
            anisotropy_angle_x=self.anisotropy_angle_x,
            anisotropy_angle_y=self.anisotropy_angle_y,
            anisotropy_angle_z=self.anisotropy_angle_z,
            drift_terms=self.drift_terms,
            point_drift=self.point_drift,
            external_drift=self.external_drift,
            external_drift_x=self.external_drift_x,
            external_drift_y=self.external_drift_y,
            functional_drift=self.functional_drift,
        )
        for kw in krige_methods_kws[self.method]:
            setup[kw] = add_setup[kw]
        input_kw = self._dimensionality_check(x)
        input_kw.update(setup)
        input_kw[self.val_kw] = y
        self.model = krige_methods[self.method](**input_kw)

    def _dimensionality_check(self, x, ext=""):
        if self.method in ("ordinary", "universal"):
            if x.shape[1] != 2:
                raise ValueError("2d krige can use only 2d points")
            else:
                return {"x" + ext: x[:, 0], "y" + ext: x[:, 1]}
        if self.method in ("ordinary3d", "universal3d"):
            if x.shape[1] != 3:
                raise ValueError("3d krige can use only 3d points")
            else:
                return {
                    "x" + ext: x[:, 0],
                    "y" + ext: x[:, 1],
                    "z" + ext: x[:, 2],
                }

    def predict(self, x, *args, **kwargs):
        if not self.model:
            raise Exception("Not trained. Train first")
        points = self._dimensionality_check(x, ext="points")
        return self.execute(points, *args, **kwargs)[0]

    def execute(self, points, *args, **kwargs):
        points.update(dict(style="points", backend="loop"))
        if isinstance(self.model, (OrdinaryKriging, OrdinaryKriging3D)):
            points.update(dict(n_closest_points=self.n_closest_points))
        #else:
        #    utilities.log.debug("n_closest_points will be ignored for UniversalKriging")
        prediction, variance = self.model.execute(**points)
        return prediction, variance
