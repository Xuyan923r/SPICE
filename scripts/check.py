#!/usr/bin/env python3
"""
Simple health check for vLLM services started by vllm_service_init/start.sh.

It sends a minimal request to /hello on the given ports and reports whether each
endpoint responds within timeout.
"""

import argparse
import json
import os
import tempfile
import time
from typing import List

import requests


def check_endpoint(host: str, port: int, timeout: float, payload_path: str) -> dict:
    url = f"http://{host}:{port}/hello"
    params = {"name": payload_path}
    start = time.time()
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        elapsed = time.time() - start
        return {
            "port": port,
            "ok": resp.ok,
            "status": resp.status_code,
            "elapsed_sec": round(elapsed, 3),
            "error": None if resp.ok else resp.text[:200],
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            "port": port,
            "ok": False,
            "status": None,
            "elapsed_sec": round(elapsed, 3),
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1", help="vLLM host (default: 127.0.0.1)")
    parser.add_argument(
        "--ports",
        default="5000,5001,5002,5003",
        help="Comma-separated list of ports to probe (default: 5000,5001,5002,5003)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="HTTP timeout in seconds for each probe (default: 10)",
    )
    args = parser.parse_args()

    ports: List[int] = [int(p.strip()) for p in args.ports.split(",") if p.strip()]

    # Create a minimal temporary JSON payload; server expects a file path.
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
        json.dump([], tmp)
        payload_path = tmp.name

    print(f"Checking vLLM endpoints on host {args.host} with payload {payload_path} ...")
    results = [check_endpoint(args.host, port, args.timeout, payload_path) for port in ports]

    for res in results:
        status = "OK" if res["ok"] else "FAIL"
        print(
            f"Port {res['port']}: {status} "
            f"(status={res['status']}, elapsed={res['elapsed_sec']}s, error={res['error']})"
        )

    # Clean up the temp file
    try:
        os.remove(payload_path)
    except OSError:
        pass


if __name__ == "__main__":
    main()
