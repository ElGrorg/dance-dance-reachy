import numpy as np
import math
import pytest
from src.utils import calculate_angle

def test_right_angle():
    """Tests a 90-degree angle."""
    a = np.array([1, 0])
    b = np.array([0, 0])
    c = np.array([0, 1])
    assert calculate_angle(a, b, c) == pytest.approx(math.pi / 2)

def test_straight_angle():
    """Tests a 180-degree angle."""
    a = np.array([1, 0])
    b = np.array([0, 0])
    c = np.array([-1, 0])
    assert calculate_angle(a, b, c) == pytest.approx(math.pi)

def test_zero_angle():
    """Tests a 0-degree angle."""
    a = np.array([1, 0])
    b = np.array([0, 0])
    c = np.array([1, 0])
    assert calculate_angle(a, b, c) == pytest.approx(0.0)

def test_collinear_point():
    """Tests with a zero-length vector (b-a)."""
    a = np.array([0, 0])
    b = np.array([0, 0])
    c = np.array([1, 1])
    assert calculate_angle(a, b, c) == pytest.approx(0.0)

def test_3d_angle():
    """Tests a 90-degree angle in 3D space."""
    a = np.array([1, 0, 0])
    b = np.array([0, 0, 0])
    c = np.array([0, 1, 0])
    assert calculate_angle(a, b, c) == pytest.approx(math.pi / 2)