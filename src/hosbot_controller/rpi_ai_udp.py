import socket
import cv2

# Server Configuration
SERVER_IP = "172.29.146.82"  # Replace with actual server IP
UDP_PORT = 5000
PACKET_SIZE = 60000  # Safe UDP packet size

# Create UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Open webcam
cap = cv2.VideoCapture(0)

print(f"[UDP] Streaming webcam to {SERVER_IP}:{UDP_PORT}")

frame_id = 0  # Track frame numbers

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Encode frame as JPEG
    _, encoded_frame = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    encoded_bytes = encoded_frame.tobytes()

    # Split frame into smaller UDP packets
    total_packets = len(encoded_bytes) // PACKET_SIZE + 1

    for i in range(total_packets):
        start = i * PACKET_SIZE
        end = start + PACKET_SIZE
        packet_data = encoded_bytes[start:end]

        # Send frame ID, packet number, and total packets
        header = f"{frame_id},{i},{total_packets}".encode()
        sock.sendto(header + b"||" + packet_data, (SERVER_IP, UDP_PORT))

    frame_id += 1  # Increment frame number

    # Real-time delay
    cv2.waitKey(1)

# Send END signal
sock.sendto(b"END_OF_FILE", (SERVER_IP, UDP_PORT))
print("[UDP] Webcam stream ended.")

cap.release()
sock.close()
