"""Integration tests for pipelines — require model weights.

These tests are marked as slow and skipped in CI.
"""

import pytest


@pytest.mark.slow
class TestTextToVideoPipeline:
    def test_generate(self):
        pytest.skip("Requires model weights")


@pytest.mark.slow
class TestImageToVideoPipeline:
    def test_generate(self):
        pytest.skip("Requires model weights")
