#!/usr/bin/env python3

"""

This is just for validating yolo_tcp_server.py from:
 - another machine or 
 - from inside the Nano container.

Measures the round-trip latency including:
 - JPEG encoding (client)
 - network transfer
 - JPEG decoding (Nano)
 - YOLO inference
 - JSON serialization
 - network transfer back

So it approximates the real robot runtime latency, not just raw inference.

See https://chatgpt.com/s/t_69ac38f6cb3881919810a636f657f0e0

"""

import json
import socket
import struct
import time
import cv2


SERVER_HOST = "jetson.local"  # Jetson Nano "host" IP address (not container)
SERVER_PORT = 5001
REQUESTS = 20

IMAGE_PATH = "../media/duckies_2_480x480.jpg"

def recv_exact(sock, n):
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("socket closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_request(sock, frame_id, jpg_bytes):
    header = {
        "frame_id": frame_id,
        "timestamp_ns": time.time_ns(),
        "encoding": "jpeg",
        "payload_size": len(jpg_bytes),
    }

    hdr = json.dumps(header).encode("utf-8")

    sock.sendall(struct.pack(">I", len(hdr)))
    sock.sendall(hdr)
    sock.sendall(jpg_bytes)


def recv_response(sock):
    hdr_len = struct.unpack(">I", recv_exact(sock, 4))[0]
    data = recv_exact(sock, hdr_len)
    return json.loads(data.decode("utf-8"))


def main():

    print(f"Loading image from {IMAGE_PATH}...")

    img = cv2.imread(IMAGE_PATH)
    ok, enc = cv2.imencode(".jpg", img)
    jpg = enc.tobytes()

    print(f"Connecting to {SERVER_HOST}:{SERVER_PORT} and sending {REQUESTS} requests...")

    timings = []
    last_resp = None

    with socket.create_connection((SERVER_HOST, SERVER_PORT), timeout=10) as sock:

        for i in range(REQUESTS):

            t0 = time.time()

            send_request(sock, i, jpg)
            resp = recv_response(sock)

            t1 = time.time()

            timings.append((t1 - t0) * 1000.0)  # ms
            last_resp = resp

    print("Last response:")
    print(json.dumps(last_resp, indent=2))

    # Print all timings after measurement
    for i, t in enumerate(timings):
        print(f"req {i+1:02d}: {t:6.1f} ms")

    # Compute average over the last avg_start samples to skip any initial warmup / startup overhead
    avg_start = REQUESTS // 4
    avg_ms = sum(timings[avg_start:]) / len(timings[avg_start:])

    print(f"\nAverage latency over the last {avg_start}..{REQUESTS} requests: {avg_ms:.1f} ms\n")


if __name__ == "__main__":
    main()
