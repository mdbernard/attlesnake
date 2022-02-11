"""Implements quaternion (Euler parameter)."""

from typing import Optional

import numpy as np

import attlesnake.parameter.base as base
import attlesnake.parameter.direction_cosine_matrix as direction_cosine_matrix
import attlesnake.parameter.principal_rotation_vector as principal_rotation_vector


class Quaternion(base.BaseAttitudeParameter):
    """Quaternion attitude parameter."""

    def __init__(
        self,
        qs: Optional[float] = None,
        qv1: Optional[float] = None,
        qv2: Optional[float] = None,
        qv3: Optional[float] = None
    ) -> None:
        """Initialize the quaternion."""
        self.s = qs
        self.v1 = qv1
        self.v2 = qv2
        self.v3 = qv3

    def __repr__(self) -> str:
        return (
            "(s, v1, v2, v3) = "
            f"({self.s:.4f}, {self.v1:.4f}, {self.v2:.4f}, {self.v3:.4f})"
        )

    def _apply_constraint(self) -> None:
        """Enforce the unit quaternion constraint."""
        self.s, self.v1, self.v2, self.v3 = self.q/np.linalg.norm(self.q)

    @property
    def q(self) -> np.ndarray:
        """Scalar-first quaternion as numpy array."""
        return np.array([self.s, self.v1, self.v2, self.v3])

    @property
    def v(self) -> np.ndarray:
        """Vector component of quaternion as numpy array."""
        return np.array([self.v1, self.v2, self.v3])

    @classmethod
    def from_dcm(cls, dcm: "direction_cosine_matrix.DCM") -> "Quaternion":
        """Sheppard's method to initialize quaternion from DCM."""
        C = dcm.array

        s_squared = 0.25*(1 + np.trace(C))
        v1_squared = 0.25*(1 + 2*C[0,0] - np.trace(C))
        v2_squared = 0.25*(1 + 2*C[1,1] - np.trace(C))
        v3_squared = 0.25*(1 + 2*C[2,2] - np.trace(C))

        sv1 = (C[1,2] - C[2,1])/4
        sv2 = (C[2,0] - C[0,2])/4
        sv3 = (C[0,1] - C[1,0])/4

        v1v2 = (C[0,1] + C[1,0])/4
        v3v1 = (C[2,0] + C[0,2])/4
        v2v3 = (C[1,2] + C[2,1])/4

        squares = [s_squared, v1_squared, v2_squared, v3_squared]
        imax = np.argmax(squares)

        if imax == 0:
            s = s_squared**0.5
            v1 = sv1/s
            v2 = sv2/s
            v3 = sv3/s
        elif imax == 1:
            v1 = v1_squared**0.5
            s = sv1/v1
            v2 = v1v2/v1
            v3 = v3v1/v1
        elif imax == 2:
            v2 = v2_squared**0.5
            s = sv2/v2
            v1 = v1v2/v2
            v3 = v2v3/v2
        else:
            v3 = v3_squared**0.5
            s = sv3/v3
            v1 = v3v1/v3
            v2 = v2v3/v3

        if s < 0:
            s *= -1
            v1 *= -1
            v2 *= -1
            v3 *= -1

        quat = cls(s, v1, v2, v3)
        quat._apply_constraint()
        return quat

    @classmethod
    def from_prv(cls, prv: "principal_rotation_vector.PRV") -> "Quaternion":
        """Initialize from principal rotation vector (PRV)."""
        qs = np.cos(prv.angle/2)
        qv1 = prv.e1*np.sin(prv.angle/2)
        qv2 = prv.e2*np.sin(prv.angle/2)
        qv3 = prv.e3*np.sin(prv.angle/2)
        quat = cls(qs, qv1, qv2, qv3)
        quat._apply_constraint()
        return quat
