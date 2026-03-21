"""Tests for sigma schedules."""

from ltx_pipelines_mlx.scheduler import DISTILLED_SIGMAS, STAGE_2_SIGMAS, get_sigma_schedule


class TestScheduler:
    def test_distilled_has_9_values_for_8_steps(self):
        # 9 sigma values = 8 denoising steps (pairs of consecutive values)
        assert len(DISTILLED_SIGMAS) == 9

    def test_stage_2_has_4_values_for_3_steps(self):
        # 4 sigma values = 3 denoising steps
        assert len(STAGE_2_SIGMAS) == 4

    def test_distilled_decreasing(self):
        for i in range(len(DISTILLED_SIGMAS) - 1):
            assert DISTILLED_SIGMAS[i] > DISTILLED_SIGMAS[i + 1]

    def test_stage_2_decreasing(self):
        for i in range(len(STAGE_2_SIGMAS) - 1):
            assert STAGE_2_SIGMAS[i] > STAGE_2_SIGMAS[i + 1]

    def test_distilled_starts_at_1(self):
        assert DISTILLED_SIGMAS[0] == 1.0

    def test_distilled_ends_at_0(self):
        assert DISTILLED_SIGMAS[-1] == 0.0

    def test_stage_2_ends_at_0(self):
        assert STAGE_2_SIGMAS[-1] == 0.0

    def test_get_schedule_truncate(self):
        sigmas = get_sigma_schedule("distilled", num_steps=4)
        assert len(sigmas) == 4
