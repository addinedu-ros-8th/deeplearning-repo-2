import socket
import threading

# Server Configuration
UDP_RECEIVE_PORT = 5001  # Port to receive video from external sender
UDP_FORWARD_PORT = 7001  # Internal port to send video to clients
BUFFER_SIZE = 65536  # Max UDP packet size

# Listens for incoming UDP video and forwards it to internal clients
def udp_video_server():
    receive_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receive_socket.bind(("0.0.0.0", UDP_RECEIVE_PORT))
    
    forward_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # Forward socket
    print(f"[UDP] Server listening for video on 0.0.0.0:{UDP_RECEIVE_PORT}")
    print(f"[UDP] Forwarding video to internal clients on 127.0.0.1:{UDP_FORWARD_PORT}")

    while True:
        try:
            data, addr = receive_socket.recvfrom(BUFFER_SIZE)
            
            if data == b"END_OF_FILE":
                print(f"[UDP] Video stream ended from {addr}")
                continue

            # Forward the received video data to internal clients
            forward_socket.sendto(data, ("127.0.0.1", UDP_FORWARD_PORT))

        except Exception as e:
            print(f"[UDP] Error: {e}")

# Start the UDP video forwarding server
udp_thread = threading.Thread(target=udp_video_server, daemon=True)
udp_thread.start()

# Keep the main thread alive
try:
    while True:
        pass
except KeyboardInterrupt:
    print("\nServer shutting down.")