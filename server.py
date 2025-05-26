import socket
#from http.cookiejar import debug

import cv2
import numpy as np
import json
import struct
import time
import os
import shutil

from FindPeople import find_people_in_image
from FindHairType import predict_hair_type
from ClothingColor import ClothingColorAnalyzer
from NewYOLO100ClothingFinder import detect_fashion_items

HOST = '127.0.0.1'  # Localhost
PORT = 65432  # Arbitrary port


# Create a TCP/IP socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    # Bind the socket to the address and port
    s.bind((HOST, PORT))

    # Listen for incoming connections
    s.listen()
    print(f"Server listening on {HOST}:{PORT}")


    def receive_image(conn):
        start_time = time.time()

        # 1. Receive the 4-byte length
        length_buf = conn.recv(4)
        if not length_buf:
            return None

        length_buf = bytearray(length_buf)
        data_length = struct.unpack('<I', length_buf)[0]
        # 2. Receive the image bytes
        image_data = b''
        while len(image_data) < data_length:
            packet = conn.recv(data_length - len(image_data))
            if not packet:
                return None
            image_data += packet
        # 3. Decode the image using OpenCV
        np_arr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        end_time = time.time()
        print(f"Time to receive & decode image: {(end_time - start_time) * 1000:.2f} ms")

        return img


    def close():
        print("Closing server...")
        s.close()

        folder_to_remove = "unity_findPeople"
        if os.path.exists(folder_to_remove):
            shutil.rmtree(folder_to_remove)
            print(f"Removed folder: {folder_to_remove}")

        print("Server closed.")

    # Accept a connection
    while True:
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")


            while True:
                data = conn.recv(1024)
                print(f"Got {len(data)} bytes for the command: {data}")# Receive command
                if not data:
                    break

                command = data.decode('utf-8')
                print(f"Received command: {command}")

                if command.lower() == "shutdown":
                    close()
                    exit()


                elif command.lower() == "imagesend":
                    img = receive_image(conn)
                    if img is not None:
                        # Get detection results
                        people_found, cropped_images = find_people_in_image(img, confidence_threshold=0.5)
                        # Send person count
                        conn.sendall(people_found.to_bytes(4, 'big'))
                        conn.send(b'')  # Force flush (works on some socket implementations)
                        # Send each cropped image
                        for person_img in cropped_images:
                            # Encode image to JPEG bytes
                            _, buffer = cv2.imencode('.jpg', person_img)
                            image_bytes = buffer.tobytes()
                            # Send image size and data
                            conn.sendall(len(image_bytes).to_bytes(4, 'big'))
                            conn.sendall(image_bytes)
                            conn.send(b'')
                        print(f"Sent {people_found} people back to Unity")
                    else:
                        print("Failed to receive image data")
                elif command.lower() == "peoplesend":
                    img = receive_image(conn)
                    if img is not None:

                        print("Image received")
                        cv2.imwrite('new_image.jpg', img)

                        # Get hair prediction
                        hair_result = predict_hair_type(img, confidence_level=0.4)
                        print(f"Here is: {hair_result}")



                        clothesList = detect_fashion_items(img)

                        cv2.imwrite('new_image3.jpg', img)
                        for clothes in clothesList:
                            print(clothes)
                        # Convert to JSON-serializable format
                        # Process person cropping
                        # Create combined JSON data
                        combined_data = {
                            "clothes": [{
                                "label": item["label"],
                                "confidence": float(item["confidence"]),
                                "bbox": [float(coord) for coord in item["bbox"]]
                            } for item in clothesList],
                            "hair": hair_result
                        }

                        # Serialize and send
                        json_data = json.dumps(combined_data)
                        header = f"{len(json_data):<10}".encode('utf-8')
                        conn.send(header + json_data.encode('utf-8'))
                        print(f"Sent {len(clothesList)} clothes detections and {hair_result} to Unity")

                        #croppedPerson = crop_largest_person(img)
                        #print("Person cropped")

                        analyzer = ClothingColorAnalyzer()
                        color_results = analyzer.analyze_clothing_colors(img)
                        for item in color_results:
                            print(f"Found {item['label']} with color {item['hex']}")

                        color_data = {
                            "colors": [{
                                "label": item["label"],
                                "hex": item["hex"],
                                "rgb": [float(c) for c in item["rgb"]]
                            } for item in color_results]
                        }
                        color_json = json.dumps(color_data)
                        color_header = f"{len(color_json):<10}".encode('utf-8')
                        conn.send(color_header + color_json.encode('utf-8'))
                        print(f"Sent {len(color_results)} color entries")

                        # pose method HERE!
                        #posedPerson = process_image(croppedPerson)


                    else:
                        print("Failed to receive image data")

