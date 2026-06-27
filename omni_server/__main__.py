"""``python -m omni_server`` / ``omni-server`` entrypoint.

Launches uvicorn on the configured host/port. Honors all ``OMNI_*`` settings via
``omni_server.config``. For multi-worker or TLS-terminating deployments, run
uvicorn/gunicorn directly against ``omni_server.main:app`` instead.
"""

from __future__ import annotations

import uvicorn

from omni_server.config import get_settings
from omni_server.main import configure_logging


def main() -> None:
    configure_logging()
    settings = get_settings()
    # Single worker on purpose: the GPU pipeline is a process-local singleton and
    # is serialised by an in-process lock. Scale out with multiple processes only
    # if each has its own GPU.
    uvicorn.run(
        "omni_server.main:app",
        host=settings.host,
        port=settings.port,
        workers=1,
    )


if __name__ == "__main__":
    main()
