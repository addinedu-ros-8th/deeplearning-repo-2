import socket
import threading
import numpy as np
import cv2

# Server Configuration
UDP_IP = "0.0.0.0"  # Listen on all available network interfaces
UDP_PORT = 5000  # UDP for video streaming
BUFFER_SIZE = 65536  # Max UDP packet size
FORWARD_IP = "127.0.0.1"  # Forward video data to another server
FORWARD_PORT = 7000  # Forward video data to another server's port

# UDP Server for Video Streaming (Handles Both Live and Saved Video)
def udp_video_server():
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.bind((UDP_IP, UDP_PORT))
    print(f"[UDP] Server listening for video on {UDP_IP}:{UDP_PORT}")

    frame_buffer = {}  # Buffer for reassembling frames

    while True:
        try:
            data, addr = udp_socket.recvfrom(BUFFER_SIZE)
            if data == b"END_OF_FILE":
                print("[UDP] Video stream ended.")
                continue

            # Extract metadata: "frame_id,packet_num,total_packets||data"
            try:
                header, packet_data = data.split(b"||", 1)
                frame_id, packet_num, total_packets = map(int, header.decode().split(","))
            except ValueError:
                print("[UDP] Corrupt packet received, skipping...")
                continue

            # Store packet in buffer
            if frame_id not in frame_buffer:
                frame_buffer[frame_id] = [None] * total_packets
            frame_buffer[frame_id][packet_num] = packet_data

            # If full frame is received, forward it to another server
            if None not in frame_buffer[frame_id]:
                full_frame_data = b"".join(frame_buffer[frame_id])

                # Forward full frame to another server
                udp_socket.sendto(full_frame_data, (FORWARD_IP, FORWARD_PORT))
                print(f"[UDP] Forwarded frame {frame_id} to {FORWARD_IP}:{FORWARD_PORT}")

                del frame_buffer[frame_id]  # Free memory

        except Exception as e:
            print(f"[UDP] Error: {e}")

    udp_socket.close()

# Start UDP video server in a separate thread
udp_thread = threading.Thread(target=udp_video_server, daemon=True)
udp_thread.start()

# Keep the main thread alive
try:
    while True:
        pass
except KeyboardInterrupt:
    print("\nServer shutting down.")
