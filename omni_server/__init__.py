"""omni-server — standalone OmniParser v2 HTTP service.

A self-contained FastAPI service that turns a screenshot into a structured list
of UI elements (plus a Set-of-Marks annotated image), intended to run on a
machine with an NVIDIA GPU. Consumers talk to it only over HTTP — see the wire
contract in ``omni_server.schemas``.
"""

__version__ = "0.1.0"
