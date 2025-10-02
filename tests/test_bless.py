import pytest
from datetime import datetime, timedelta
from rgpycrumbs.parsers.bless import BLESS_LOG, parse_bless_time, parse_bless_log_line


class TestBlessLogParsing:
    def test_valid_log_line_parsing(self):
        log_line = "[2024-10-28T18:58:24Z] Some log message here"
        result = parse_bless_log_line(log_line)
        assert result == ("2024-10-28T18:58:24Z", "Some log message here")

    def test_valid_log_line_with_empty_data(self):
        log_line = "[2024-10-28T18:58:24Z] "
        result = parse_bless_log_line(log_line)
        assert result == ("2024-10-28T18:58:24Z", "")

    def test_valid_log_line_with_no_space_after_timestamp(self):
        log_line = "[2024-10-28T18:58:24Z]No space"
        result = parse_bless_log_line(log_line)
        assert result == ("2024-10-28T18:58:24Z", "No space")

    def test_invalid_log_line_no_brackets(self):
        log_line = "2024-10-28T18:58:24Z Some log message"
        result = parse_bless_log_line(log_line)
        assert result is None

    def test_invalid_log_line_empty(self):
        log_line = ""
        result = parse_bless_log_line(log_line)
        assert result is None

    def test_invalid_log_line_only_timestamp(self):
        log_line = "[2024-10-28T18:58:24Z]"
        result = parse_bless_log_line(log_line)
        assert result is None  # Missing space and logdata

    def test_regex_direct_match(self):
        log_line = "[2024-10-28T18:58:24Z] Test data"
        match = BLESS_LOG.match(log_line)
        assert match is not None
        assert match.group("timestamp") == "2024-10-28T18:58:24Z"
        assert match.group("logdata") == "Test data"


class TestBlessTimeParsing:
    def test_valid_timestamp_parsing(self):
        timestamp_str = "2024-10-28T18:58:24Z"
        result = parse_bless_time(timestamp_str)
        expected = datetime(2024, 10, 28, 18, 58, 24)
        assert result == expected

    def test_time_difference_calculation(self):
        start_str = "2024-10-28T18:58:21Z"
        end_str = "2024-10-28T18:58:24Z"
        start_time = parse_bless_time(start_str)
        end_time = parse_bless_time(end_str)
        difference = end_time - start_time
        assert difference == timedelta(seconds=3)

    def test_invalid_timestamp_format(self):
        timestamp_str = "2024-10-28 18:58:24"  # Missing T and Z
        with pytest.raises(ValueError):
            parse_bless_time(timestamp_str)

    def test_timestamp_with_milliseconds(self):
        # Original format doesn't support ms, but test if it fails gracefully
        timestamp_str = "2024-10-28T18:58:24.123Z"
        with pytest.raises(ValueError):
            parse_bless_time(timestamp_str)

    def test_empty_timestamp(self):
        timestamp_str = ""
        with pytest.raises(ValueError):
            parse_bless_time(timestamp_str)

    def test_non_iso_timestamp(self):
        timestamp_str = "28/10/2024 18:58:24"
        with pytest.raises(ValueError):
            parse_bless_time(timestamp_str)
