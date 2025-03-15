import socket

SERVER_IP = "192.168.1.100"  # Replace with actual server IP
TCP_PORT = 5000

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((SERVER_IP, TCP_PORT))

message = "Hello from TCP client!"
client.sendall(message.encode())

data = client.recv(1024)
print(f"Received from server: {data.decode()}")

client.close()