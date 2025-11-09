import socket, struct, time

HOST, PORT = "127.0.0.1", 22081

# def sendmessage(sock, msg_id, payload=b""):
#     total_len = len(payload) + 4
#     header = b"VAST" + struct.pack("<Q", total_len) + struct.pack("<I", msg_id)
#     msg = header + payload
#     print("Sending:", msg.hex())
#     sock.sendall(msg)

# with socket.create_connection((HOST, PORT), timeout=5) as s:
#     print("Connected to VAST")
#     # GETINFO = 6 (guess)
#     sendmessage(s, 6)
#     time.sleep(0.1)
#     try:
#         data = s.recv(4096)
#         print("Response:", data)
#     except Exception as e:
#         print("recv error:", e)

data =  b'\x01p\x17\x00\x00\x01p\x17\x00\x00\x01x\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x14@\x02\x00\x00\x00\x00\x00\x00\x14@\x02\x00\x00\x00\x00\x00\x00I@\x01\x10\x00\x00\x00\x01\x10\x00\x00\x00\x01\x10\x00\x00\x00\x04\xb5\x0c\x00\x00\x04\xc7\x0b\x00\x00\x04\x10\x00\x00\x00\x01\x05\x00\x00\x00'

if not data.startswith(b"VAST"):
    total_len = struct.unpack("<Q", data[4:12])[0]
    msg_type = struct.unpack("<I", data[12:16])[0]
    payload_bytes = data[16:]
    print(f"total_len: {total_len}, msg_type: {msg_type}, payload len: {len(payload_bytes)}")
    print(f"payload: {payload_bytes.hex()}")