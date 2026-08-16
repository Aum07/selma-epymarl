"""Repository-wide test guards.

**No test in this repository may touch the network.** The SELMA Stage D provider
adapter is exercised exclusively through injected fake clients; this guard makes
that a property of the test run rather than a convention, so a future test that
accidentally constructs a real client fails loudly instead of spending money.

The guard patches the connect entry points rather than socket construction, so
libraries that merely create a socket object (or use socketpairs internally)
still work, while any attempt to reach a peer raises.
"""

import socket

import pytest


class NetworkAccessDenied(RuntimeError):
    """A test attempted a real network connection."""


@pytest.fixture(autouse=True, scope="session")
def _forbid_network_access():
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_create_connection = socket.create_connection

    def denied(*args, **kwargs):
        raise NetworkAccessDenied(
            "the test suite is forbidden from making network connections; "
            "provider tests must use an injected fake client"
        )

    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    socket.create_connection = denied
    try:
        yield
    finally:
        socket.socket.connect = real_connect
        socket.socket.connect_ex = real_connect_ex
        socket.create_connection = real_create_connection
