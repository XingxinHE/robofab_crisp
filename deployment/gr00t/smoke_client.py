"""Smoke-test a running GR00T server without ROS/CRISP."""

from __future__ import annotations

from deployment.gr00t.smoke_http_client import main


if __name__ == "__main__":
    raise SystemExit(main())
