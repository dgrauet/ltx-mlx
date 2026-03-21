"""Tests for the Euler denoising math."""

import mlx.core as mx

from ltx_pipelines_mlx.denoise import euler_step


class TestEulerStep:
    def test_zero_sigma_returns_x0(self):
        x = mx.ones((1, 4, 8))
        x0 = mx.zeros((1, 4, 8))
        result = euler_step(x, x0, sigma=0.0, sigma_next=0.0)
        assert mx.allclose(result, x0).item()

    def test_step_towards_x0(self):
        x = mx.ones((1, 4, 8))
        x0 = mx.zeros((1, 4, 8))
        # Going from sigma=1.0 to sigma=0.0 should give x0
        result = euler_step(x, x0, sigma=1.0, sigma_next=0.0)
        assert mx.allclose(result, x0, atol=1e-5).item()

    def test_partial_step(self):
        x = mx.ones((1, 4, 8)) * 2.0
        x0 = mx.zeros((1, 4, 8))
        # Going from sigma=1.0 to sigma=0.5: should move halfway
        result = euler_step(x, x0, sigma=1.0, sigma_next=0.5)
        expected = mx.ones((1, 4, 8))  # halfway between 2 and 0
        assert mx.allclose(result, expected, atol=1e-5).item()
