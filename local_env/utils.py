"""
Shared utilities for logging, subprocess management, and downloads.
"""

import subprocess
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any


class Logger:
    """Logging utility that writes to console and file."""

    PREFIX = "[SAM3DObjects]"

    def __init__(self, log_file: Path):
        self.log_file = log_file
        # Initialize log file
        with open(log_file, 'w') as f:
            f.write(f"SAM3D Installation Log - {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")

    def info(self, message: str):
        """Log info message to console and file."""
        print(f"{self.PREFIX} {message}")
        self._write_to_file("INFO", message)

    def warning(self, message: str):
        """Log warning message to console and file."""
        print(f"{self.PREFIX} WARNING: {message}")
        self._write_to_file("WARN", message)

    def error(self, message: str):
        """Log error message to console and file."""
        print(f"{self.PREFIX} ERROR: {message}")
        self._write_to_file("ERROR", message)

    def success(self, message: str):
        """Log success message with checkmark."""
        # Use ASCII-safe checkmark for Windows compatibility
        print(f"{self.PREFIX} [OK] {message}")
        self._write_to_file("OK", message)

    def _write_to_file(self, level: str, message: str):
        """Write message to log file."""
        with open(self.log_file, 'a') as f:
            timestamp = datetime.now().strftime('%H:%M:%S')
            f.write(f"[{timestamp}] [{level}] {message}\n")

    def log_subprocess(self, step_name: str, stdout: str = "", stderr: str = ""):
        """Log subprocess output to file only."""
        with open(self.log_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"{step_name}\n")
            f.write(f"{'='*80}\n")
            if stdout:
                f.write(f"STDOUT:\n{stdout}\n")
            if stderr:
                f.write(f"STDERR:\n{stderr}\n")

    def run_logged(
        self,
        cmd: list,
        step_name: str = None,
        check: bool = True,
        **kwargs
    ) -> subprocess.CompletedProcess:
        """
        Run a subprocess with output logged to file.

        Args:
            cmd: Command and arguments
            step_name: Description for logging
            check: Raise exception on non-zero exit
            **kwargs: Additional subprocess.run arguments

        Returns:
            CompletedProcess result
        """
        step_name = step_name or ' '.join(str(c) for c in cmd[:3])

        # Force capture_output
        kwargs['capture_output'] = True
        kwargs['text'] = True

        try:
            result = subprocess.run(cmd, **kwargs)
            self.log_subprocess(f"{step_name} - EXIT {result.returncode}",
                              result.stdout, result.stderr)

            if check and result.returncode != 0:
                raise subprocess.CalledProcessError(
                    result.returncode, cmd, result.stdout, result.stderr
                )

            return result

        except subprocess.CalledProcessError as e:
            self.log_subprocess(f"{step_name} - FAILED",
                              getattr(e, 'stdout', ''),
                              getattr(e, 'stderr', ''))
            self.error(f"{step_name} failed!")
            self.error(f"Check logs at: {self.log_file}")

            # Show last bit of output for debugging
            if hasattr(e, 'stdout') and e.stdout:
                print(f"\nLast output:\n{e.stdout[-500:]}")
            if hasattr(e, 'stderr') and e.stderr:
                print(f"\nError output:\n{e.stderr[-500:]}")
            raise


def validate_url(url: str, timeout: int = 10) -> Dict[str, Any]:
    """
    Validate that a URL is accessible.

    Args:
        url: URL to validate
        timeout: Request timeout in seconds

    Returns:
        Dict with 'valid', 'status_code', 'content_length', 'error' keys
    """
    result = {
        'valid': False,
        'url': url,
        'status_code': None,
        'content_length': None,
        'error': None
    }

    # Try HEAD request first
    request = urllib.request.Request(url, method='HEAD')

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result['status_code'] = response.status
            result['valid'] = 200 <= response.status < 300
            result['content_length'] = response.headers.get('Content-Length')
        return result

    except urllib.error.HTTPError as e:
        result['status_code'] = e.code
        result['error'] = f'HTTP {e.code}: {e.reason}'

        # Try GET with Range header if HEAD not allowed
        if e.code == 405:
            try:
                request = urllib.request.Request(url)
                request.add_header('Range', 'bytes=0-0')
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    result['status_code'] = response.status
                    result['valid'] = True
                    result['error'] = None
            except Exception:
                pass

        return result

    except urllib.error.URLError as e:
        result['error'] = f'URL error: {e.reason}'
        return result

    except Exception as e:
        result['error'] = f'Unexpected error: {e}'
        return result


def download_file(url: str, dest: Path, logger: Optional[Logger] = None) -> None:
    """
    Download a file from URL to destination.

    Args:
        url: URL to download from
        dest: Destination path
        logger: Optional logger for progress messages
    """
    if logger:
        logger.info(f"Downloading {dest.name}...")

    urllib.request.urlretrieve(url, dest)

    if logger:
        size_kb = dest.stat().st_size / 1024
        logger.info(f"Downloaded {size_kb:.1f} KB")
