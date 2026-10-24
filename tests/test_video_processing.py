"""
test_video_processing.py
Tests for video file validation, metadata, and frame sampling logic.
No real video files are needed; we test with minimal synthetic or invalid inputs.
"""

import os
import tempfile
import numpy as np
import pytest

from src.video_processing import (
    validate_video_file,
    SUPPORTED_EXTENSIONS,
    MAX_FILE_SIZE_MB,
    MIN_RESOLUTION,
    MIN_DURATION_SEC,
    VideoMetadata,
)


class TestValidateVideoFile:
    """Tests for the file-level validation function."""

    def test_nonexistent_file_returns_error(self):
        errors = validate_video_file("/nonexistent/path/video.mp4")
        assert len(errors) > 0
        assert any("not found" in e.lower() for e in errors)

    def test_unsupported_extension_returns_error(self, tmp_path):
        bad_file = tmp_path / "video.xyz"
        bad_file.write_bytes(b"fake content")
        errors = validate_video_file(str(bad_file))
        assert len(errors) > 0
        assert any("unsupported" in e.lower() or "extension" in e.lower() for e in errors)

    def test_supported_extension_passes_file_check(self, tmp_path):
        # A tiny file with a valid extension passes the extension check
        # (content check happens later)
        valid_file = tmp_path / "video.mp4"
        valid_file.write_bytes(b"x" * 100)
        errors = validate_video_file(str(valid_file))
        # Should not contain extension error; may contain no errors or size-related ones
        assert not any("unsupported" in e.lower() for e in errors)

    def test_all_supported_extensions_accepted(self, tmp_path):
        for ext in SUPPORTED_EXTENSIONS:
            f = tmp_path / f"video{ext}"
            f.write_bytes(b"x" * 100)
            errors = validate_video_file(str(f))
            assert not any("unsupported" in e.lower() for e in errors), \
                f"Extension {ext} should be supported"

    def test_case_insensitive_extension(self, tmp_path):
        upper_file = tmp_path / "VIDEO.MP4"
        upper_file.write_bytes(b"x" * 100)
        errors = validate_video_file(str(upper_file))
        assert not any("unsupported" in e.lower() for e in errors)

    def test_oversized_file_returns_error(self, tmp_path):
        """Simulate a file that exceeds the size limit by patching os.path.getsize."""
        import src.video_processing as vp
        big_file = tmp_path / "big.mp4"
        big_file.write_bytes(b"x")

        original_getsize = os.path.getsize
        try:
            os.path.getsize = lambda p: (MAX_FILE_SIZE_MB + 10) * 1024 * 1024
            errors = vp.validate_video_file(str(big_file))
            assert any("size" in e.lower() or "limit" in e.lower() for e in errors)
        finally:
            os.path.getsize = original_getsize


class TestSupportedFormats:
    def test_mp4_in_supported_set(self):
        assert ".mp4" in SUPPORTED_EXTENSIONS

    def test_mov_in_supported_set(self):
        assert ".mov" in SUPPORTED_EXTENSIONS

    def test_all_extensions_start_with_dot(self):
        for ext in SUPPORTED_EXTENSIONS:
            assert ext.startswith("."), f"Extension {ext!r} should start with '.'"


class TestVideoMetadata:
    """Test the VideoMetadata dataclass."""

    def test_metadata_fields_accessible(self):
        meta = VideoMetadata(
            filepath="/tmp/test.mp4",
            width=1280,
            height=720,
            fps=30.0,
            frame_count=900,
            duration_sec=30.0,
            file_size_mb=15.0,
            codec="avc1",
        )
        assert meta.width == 1280
        assert meta.height == 720
        assert meta.fps == 30.0
        assert meta.duration_sec == 30.0
        assert meta.codec == "avc1"

    def test_duration_calculation_consistency(self):
        fps = 25.0
        frame_count = 250
        expected_duration = frame_count / fps  # 10.0 seconds
        meta = VideoMetadata(
            filepath="x.mp4",
            width=640, height=480,
            fps=fps, frame_count=frame_count,
            duration_sec=expected_duration,
            file_size_mb=5.0,
        )
        assert abs(meta.duration_sec - 10.0) < 0.01
