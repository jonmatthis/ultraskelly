"""A free and open source markerless motion capture system for everyone 💀✨"""

__author__ = """Skelly FreeMoCap"""
__email__ = "info@ultraskelly.org"
__version__ = "v1.4.7"
__description__ = "A free and open source markerless motion capture system for everyone 💀✨"

__package_name__ = "ultraskelly"
__repo_url__ = f"https://github.com/ultraskelly/{__package_name__}/"
__repo_issues_url__ = f"{__repo_url__}issues"

from ultraskelly.system.logging_configuration.configure_logging import configure_logging
from ultraskelly.system.logging_configuration.log_levels import LogLevels

LOG_LEVEL = LogLevels.TRACE
configure_logging(LOG_LEVEL)

