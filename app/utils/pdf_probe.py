from __future__ import annotations

import re
import socket
from ipaddress import IPv6Address, ip_address
from urllib.parse import urljoin, urlsplit

import aiohttp
from anyio import fail_after

_PDF_HEADER = b"%PDF-"
_MAX_REDIRECTS = 3
_PROBE_TIMEOUT_SECONDS = 10


def _public_socket(addr_info: tuple) -> socket.socket:
    family, kind, protocol, _, sockaddr = addr_info
    address = ip_address(sockaddr[0])
    if isinstance(address, IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    if not address.is_global or address.is_multicast:
        raise OSError("PDF URL must connect to a public address")
    return socket.socket(family, kind, protocol)


async def probe_pdf_url(url: str) -> bool:
    """Check for a PDF header using a bounded, public HTTP(S) request."""
    try:
        with fail_after(_PROBE_TIMEOUT_SECONDS):
            async with aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(socket_factory=_public_socket),
                auto_decompress=False,
                headers={"Range": "bytes=0-1023", "Accept-Encoding": "identity"},
            ) as client:
                for redirect_count in range(_MAX_REDIRECTS + 1):
                    target = urlsplit(url)
                    if (
                        target.scheme not in {"http", "https"}
                        or not target.hostname
                        or target.username is not None
                    ):
                        return False

                    async with client.get(url, allow_redirects=False) as response:
                        if response.status in {301, 302, 303, 307, 308}:
                            location = response.headers.get("Location")
                            if not location or redirect_count == _MAX_REDIRECTS:
                                return False
                            url = urljoin(url, location)
                            continue

                        if response.status not in {200, 206}:
                            return False
                        if response.status == 206 and not re.fullmatch(
                            r"bytes 0-[0-9]+/(?:[0-9]+|\*)",
                            response.headers.get("Content-Range", ""),
                            flags=re.IGNORECASE,
                        ):
                            return False
                        prefix = await response.content.readexactly(len(_PDF_HEADER))
                        return prefix == _PDF_HEADER
    except (aiohttp.ClientError, EOFError, OSError, ValueError):
        return False
    return False
