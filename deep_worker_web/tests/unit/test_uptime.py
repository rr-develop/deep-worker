"""Tests for uptime utilities."""

import pytest
import time
from app.utils import uptime


class TestUptimeFormatting:
    """Tests for uptime formatting functions."""

    def test_format_uptime_zero_seconds(self):
        """Test formatting 0 seconds."""
        assert uptime.format_uptime(0) == "0s"
        assert uptime.format_uptime(0.5) == "0s"  # Rounds down

    def test_format_uptime_seconds_only(self):
        """Test formatting seconds only (< 1 minute)."""
        assert uptime.format_uptime(1) == "1s"
        assert uptime.format_uptime(45) == "45s"
        assert uptime.format_uptime(59.9) == "59s"

    def test_format_uptime_minutes_and_seconds(self):
        """Test formatting minutes and seconds."""
        assert uptime.format_uptime(60) == "1m"
        assert uptime.format_uptime(90) == "1m 30s"
        assert uptime.format_uptime(150) == "2m 30s"
        assert uptime.format_uptime(3599) == "59m 59s"

    def test_format_uptime_hours_minutes_seconds(self):
        """Test formatting hours, minutes, and seconds."""
        assert uptime.format_uptime(3600) == "1h"
        assert uptime.format_uptime(3660) == "1h 1m"
        assert uptime.format_uptime(3920) == "1h 5m 20s"
        assert uptime.format_uptime(7385) == "2h 3m 5s"

    def test_format_uptime_days_no_seconds(self):
        """Test formatting days - seconds should not be shown when days present."""
        assert uptime.format_uptime(86400) == "1d"
        assert uptime.format_uptime(86460) == "1d 1m"  # 1 day + 1 minute
        assert uptime.format_uptime(90065) == "1d 1h 1m"  # No seconds shown
        assert uptime.format_uptime(183900) == "2d 3h 5m"  # 2 days, 3 hours, 5 minutes

    def test_format_uptime_complex_values(self):
        """Test formatting complex uptime values."""
        # 1 day, 5 hours, 23 minutes, 45 seconds
        assert uptime.format_uptime(105825) == "1d 5h 23m"
        # 7 days, 12 hours
        assert uptime.format_uptime(648000) == "7d 12h"


class TestUptimeCalculation:
    """Tests for uptime calculation."""

    def test_calculate_uptime_returns_positive_value(self):
        """Test that calculate_uptime returns positive value."""
        uptime_val = uptime.calculate_uptime()
        assert uptime_val > 0, "Uptime should be positive"

    def test_calculate_uptime_increases_over_time(self):
        """Test that uptime increases as time passes."""
        uptime1 = uptime.calculate_uptime()
        time.sleep(0.1)
        uptime2 = uptime.calculate_uptime()
        assert uptime2 > uptime1, "Uptime should increase"

    def test_set_server_start_time_affects_uptime(self):
        """Test that set_server_start_time changes calculated uptime."""
        # Set start time to 1 hour ago
        one_hour_ago = time.time() - 3600
        uptime.set_server_start_time(one_hour_ago)

        calculated = uptime.calculate_uptime()
        # Should be approximately 1 hour (3600 seconds), allow small tolerance
        assert 3599 < calculated < 3601, f"Expected ~3600s, got {calculated}"

    def test_get_uptime_data_returns_dict_with_required_fields(self):
        """Test that get_uptime_data returns dict with uptime_seconds and uptime_formatted."""
        data = uptime.get_uptime_data()

        assert isinstance(data, dict), "Should return dict"
        assert "uptime_seconds" in data, "Should have uptime_seconds key"
        assert "uptime_formatted" in data, "Should have uptime_formatted key"

        assert isinstance(data["uptime_seconds"], float), "uptime_seconds should be float"
        assert isinstance(data["uptime_formatted"], str), "uptime_formatted should be str"

        assert data["uptime_seconds"] > 0, "uptime_seconds should be positive"
        assert len(data["uptime_formatted"]) > 0, "uptime_formatted should not be empty"

    def test_get_uptime_data_consistency(self):
        """Test that uptime_formatted matches uptime_seconds in get_uptime_data."""
        data = uptime.get_uptime_data()

        # Manually format the seconds and compare
        expected_formatted = uptime.format_uptime(data["uptime_seconds"])
        assert data["uptime_formatted"] == expected_formatted, \
            f"Formatted uptime mismatch: {data['uptime_formatted']} != {expected_formatted}"
