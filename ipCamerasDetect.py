import scapy.all as scapy
import socket

# Define common IP camera ports
CAMERA_PORTS = [80, 554, 8080]

def scan_network(ip_range):
    print(f"Scanning IP range: {ip_range}")
    arp_request = scapy.ARP(pdst=ip_range)
    broadcast = scapy.Ether(dst="ff:ff:ff:ff:ff:ff")
    arp_request_broadcast = broadcast / arp_request
    answered_list = scapy.srp(arp_request_broadcast, timeout=2, verbose=False)[0]
    
    devices = []
    for element in answered_list:
        devices.append(element[1].psrc)  # Add the responding IP addresses

    return devices

def is_camera(ip):
    for port in CAMERA_PORTS:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)  # Set timeout for the connection attempt
            if sock.connect_ex((ip, port)) == 0:  # Port is open
                print(f"Camera found at {ip}:{port}")
                return True
    return False

def main():
    # Specify your local network range
    local_ip = "192.168.1.0/24"  # Adjust this according to your network
    devices = scan_network(local_ip)

    print("\nDetected Devices:")
    for device in devices:
        print(device)
        if is_camera(device):
            print(f"IP Camera detected: {device}")

if __name__ == "__main__":
    main()
