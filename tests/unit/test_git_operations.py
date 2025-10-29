"""Unit tests for git operations functions."""

import pytest
from unittest.mock import patch, MagicMock
import subprocess


class TestCreateGitTags:
    """Test create_git_tags function."""

    @patch("subprocess.run")
    def test_create_git_tags_success(self, mock_run):
        """Test successful git tag creation."""
        from src.server.server_utils import create_git_tags

        mock_run.return_value = MagicMock()

        create_git_tags("2025-10-27T12:00:00", "0.3.6", 50)

        # Should have called git tag and git push
        assert mock_run.call_count >= 2
        calls = mock_run.call_args_list
        assert "git" in str(calls[0])
        assert "tag" in str(calls[0])
        assert "fl-run-20251027T120000-v0.3.6" in str(calls[0])
        assert "push" in str(calls[1])

    @patch("subprocess.run")
    def test_create_git_tags_failure(self, mock_run):
        """Test git tag creation failure."""
        from src.server.server_utils import create_git_tags

        mock_run.side_effect = subprocess.CalledProcessError(1, ["git", "tag"])

        # Should not raise exception, just log warning
        create_git_tags("2025-10-27T12:00:00", "0.3.6", 50)