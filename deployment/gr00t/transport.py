"""GR00T inference transport clients.

The ZMQ protocol mirrors Isaac-GR00T's ``gr00t.eval.service`` implementation:
requests are Torch-serialized dicts with an endpoint name and optional data.
This keeps CRISP deployment independent from the Isaac-GR00T Python package.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any, Literal, Protocol

import json_numpy
import requests
import torch


json_numpy.patch()

GrootTransport = Literal["http", "zmq"]


class Gr00tInferenceClient(Protocol):
    def check_health(self) -> None:
        """Validate that the inference server is reachable."""

    def get_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Return an action horizon for one observation."""

    def close(self) -> None:
        """Release client resources."""


class HttpGr00tClient:
    def __init__(
        self,
        *,
        server_url: str,
        timeout_sec: float,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.session = requests.Session()

    def check_health(self) -> None:
        response = self.session.get(
            f"{self.server_url}/health",
            timeout=self.timeout_sec,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"GR00T HTTP health check failed: {response.status_code} {response.text}"
            )

    def get_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        response = self.session.post(
            f"{self.server_url}/act",
            json={"observation": observation},
            timeout=self.timeout_sec,
        )
        if response.status_code != 200:
            raise RuntimeError(
                "GR00T HTTP action request failed: "
                f"{response.status_code} {response.text[:1000]}"
            )
        return response.json()

    def close(self) -> None:
        self.session.close()


class ZmqGr00tClient:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        timeout_sec: float,
        api_token: str | None = None,
    ) -> None:
        try:
            import zmq
        except ImportError as exc:
            raise ImportError(
                "pyzmq is required for --groot-transport zmq. "
                "Run `pixi install` in robofab_crisp after pulling this change."
            ) from exc

        self.zmq = zmq
        self.host = host
        self.port = port
        self.timeout_ms = int(timeout_sec * 1000)
        self.api_token = api_token
        self.context = zmq.Context()
        self.socket = None
        self._init_socket()

    def _init_socket(self) -> None:
        if self.socket is not None:
            self.socket.close(linger=0)

        self.socket = self.context.socket(self.zmq.REQ)
        self.socket.setsockopt(self.zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(self.zmq.SNDTIMEO, self.timeout_ms)
        self.socket.setsockopt(self.zmq.LINGER, 0)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def check_health(self) -> None:
        self._call_endpoint("ping", requires_input=False)

    def get_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        return self._call_endpoint("get_action", data=observation)

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close(linger=0)
            self.socket = None
        self.context.term()

    def _call_endpoint(
        self,
        endpoint: str,
        data: dict[str, Any] | None = None,
        requires_input: bool = True,
    ) -> dict[str, Any]:
        if self.socket is None:
            self._init_socket()

        request: dict[str, Any] = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data or {}
        if self.api_token:
            request["api_token"] = self.api_token

        try:
            self.socket.send(_torch_to_bytes(request))
            response = _torch_from_bytes(self.socket.recv())
        except self.zmq.Again as exc:
            self._init_socket()
            raise TimeoutError(
                f"Timed out waiting for GR00T ZMQ endpoint {endpoint!r} "
                f"after {self.timeout_ms / 1000:.1f}s."
            ) from exc

        if "error" in response:
            raise RuntimeError(f"GR00T ZMQ server error: {response['error']}")
        return response


def make_groot_client(
    *,
    transport: GrootTransport,
    server_url: str,
    host: str,
    port: int,
    timeout_sec: float,
    api_token: str | None = None,
) -> Gr00tInferenceClient:
    if transport == "http":
        return HttpGr00tClient(server_url=server_url, timeout_sec=timeout_sec)
    if transport == "zmq":
        return ZmqGr00tClient(
            host=host,
            port=port,
            timeout_sec=timeout_sec,
            api_token=api_token,
        )
    raise ValueError(f"Unsupported GR00T transport: {transport!r}")


def _torch_to_bytes(data: dict[str, Any]) -> bytes:
    buffer = BytesIO()
    torch.save(data, buffer)
    return buffer.getvalue()


def _torch_from_bytes(data: bytes) -> dict[str, Any]:
    buffer = BytesIO(data)
    return torch.load(buffer, weights_only=False)
