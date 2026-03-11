#!/usr/bin/env python3

"""

This is just for validating yolo_tcp_server.py from:
 - another machine or 
 - from inside the Nano container.

Server code: https://github.com/slgrobotics/jetson_nano_b01/blob/main/src/yolo_tcp_server.py

See https://chatgpt.com/s/t_69ab72e92950819191c249c64f5adc5b

Usage:
 - Make sure yolo_tcp_server.py is running on the Jetson Nano (in the container).
 - Update SERVER_HOST to the correct IP address of the Jetson Nano (not the container).
 - Run this script on another machine that can reach the Jetson Nano over the network, or inside the container.
 - expected output is the JSON response from the server with inference results for the provided "duckies" image.

"""

import json
import socket
import struct
import time
import cv2

SERVER_HOST = "jetson.local"  # Jetson Nano "host" IP address (not container)
SERVER_PORT = 5001

IMAGE_PATH = "../media/duckies_2_480x480.jpg"

def recv_exact(sock, n):
    """
    @brief Receive exactly n bytes from the socket. This handles cases where recv() may return less than n bytes.
    """
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
    """
    @brief Send a single inference request to the server with the given frame ID and JPEG bytes.
    """
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
    """
    @brief Receive a single inference response from the server and return it as a Python dictionary.
    """
    n = struct.unpack(">I", recv_exact(sock, 4))[0]
    data = recv_exact(sock, n)
    return json.loads(data.decode("utf-8"))


print(f"Loading image from {IMAGE_PATH}...")

img = cv2.imread(IMAGE_PATH)
ok, enc = cv2.imencode(".jpg", img)
jpg = enc.tobytes()

print(f"Connecting to {SERVER_HOST}:{SERVER_PORT} and sending request...")

with socket.create_connection((SERVER_HOST, SERVER_PORT), timeout=10) as sock:
    send_request(sock, 1, jpg)
    resp = recv_response(sock)
    print(json.dumps(resp, indent=2))

