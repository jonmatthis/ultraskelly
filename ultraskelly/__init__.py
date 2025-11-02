"""Animatronic UltraSkelly 💀🤖"""

__author__ = """Skelly FreeMoCap"""
__email__ = "info@ultraskelly.org"
__version__ = "v0.1.0"
__description__ = "Animatronic UltraSkelly ????"

__package_name__ = "jonmatthis"
__github_username__ = "ultraskelly"
__repo_url__ = f"https://github.com/{__github_username__}/{__package_name__}/"
__repo_issues_url__ = f"{__repo_url__}issues"

from ultraskelly.system.logging_configuration.configure_logging import configure_logging
from ultraskelly.system.logging_configuration.log_levels import LogLevels

LOG_LEVEL = LogLevels.TRACE
configure_logging(LOG_LEVEL)

