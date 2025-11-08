import socket, struct, time

HOST, PORT = "127.0.0.1", 22081

def sendmessage(sock, msg_id, payload=b""):
    total_len = len(payload) + 4
    header = b"VAST" + struct.pack("<Q", total_len) + struct.pack("<I", msg_id)
    msg = header + payload
    print("Sending:", msg.hex())
    sock.sendall(msg)

with socket.create_connection((HOST, PORT), timeout=5) as s:
    print("Connected to VAST")
    # GETINFO = 6 (guess)
    sendmessage(s, 6)
    time.sleep(0.1)
    try:
        data = s.recv(4096)
        print("Response:", data)
    except Exception as e:
        print("recv error:", e)
