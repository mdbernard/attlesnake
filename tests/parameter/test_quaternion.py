"""Unit tests for Quaternion."""

import numpy as np

from attlesnake import Quaternion


def test_quaternion_composition():
    """Test basic quaternion composition."""
    # arrange
    q_FN = Quaternion(0.359211, 0.898027, 0.179605, 0.179605)
    q_NB = Quaternion(-0.377964, -0.755929, -0.377964, -0.377964)

    q_FB_expected = Quaternion(6.78844289e-01, -6.10959902e-01, -4.07306309e-01, 1.98359004e-07)

    # act
    q_FB = q_FN @ q_NB

    # assert
    np.testing.assert_allclose(q_FB.q, q_FB_expected.q)


def test_quaternion_composition_returns_minimal_path():
    """
    Test quaternion composition returns the result with the minimal
    path of the two valid results.
    """
    # arrange
    q_FB = Quaternion(0.359211, 0.898027, 0.179605, 0.179605)
    q_BN = Quaternion(0.774597, 0.258199, 0.516398, 0.258199)

    q_FN_expected = Quaternion(0.09274732, -0.83473002, -0.51011272, 0.18549593)

    # act
    q_FN = q_FB @ q_BN

    # assert
    np.testing.assert_allclose(q_FN.q, q_FN_expected.q)


def test_quaternion_inverse():
    """Test quaternion inverse."""
    # arrange
    q = Quaternion(-0.377964, 0.755929, 0.377964, 0.377964)

    q_inv_expected = Quaternion(-0.377964, -0.755929, -0.377964, -0.377964)

    # act
    q_inv = q.inverse()

    # assert
    np.testing.assert_allclose(q_inv.q, q_inv_expected.q)
