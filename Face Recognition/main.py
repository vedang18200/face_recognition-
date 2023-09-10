import face_recognition 
import cv2
import numpy as np
import csv
import os
from datetime import datetime

# Initialize video capture
video_capture = cv2.VideoCapture(0)

# Load face images and encodings
vedangs_image = face_recognition.load_image_file("resources/vedang.jpg")
vedangs_encoding = face_recognition.face_encodings(vedangs_image)[0]

jayesh_image = face_recognition.load_image_file("resources/jayass.jpg")
jayesh_encoding = face_recognition.face_encodings(jayesh_image)[0]

# Define roll numbers and GR numbers
vedangs_roll_number = "09"
vedangs_gr_number = "S2280293"

jayesh_roll_number = "34"
jayesh_gr_number = "S2287299"

# Load additional face images and encodings
arya_image = face_recognition.load_image_file("resources/arya.jpg")
arya_encoding = face_recognition.face_encodings(arya_image)[0]

bhagshree_image = face_recognition.load_image_file("resources/bhagshree.jpg")
bhagshree_encoding = face_recognition.face_encodings(bhagshree_image)[0]

# Define roll numbers and GR numbers for additional persons
arya_roll_number = "49"
arya_gr_number = "GR003"

bhagshree_roll_number = "004"
bhagshree_gr_number = "GR004"

# Define a dictionary to map names to roll numbers and GR numbers
person_info = {
    "vedang deshmukh": {"roll_number": "09", "gr_number": "S2280293"},
    "jayesh patil": {"roll_number": "34", "gr_number": "S2287299"},
    "arya sawant": {"roll_number": "49", "gr_number": "S2280277"},
    "bhagshree Sonar": {"roll_number": "", "gr_number": "S2287264"}
}


# Lists to store face information
known_face_encodings = [
    vedangs_encoding,
    jayesh_encoding,
    arya_encoding,
    bhagshree_encoding
]

known_face_names = list(person_info.keys())

students = known_face_names.copy()

# Lists to store face information
face_locations = []
face_names = []

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

# Open the CSV file for writing attendance
f = open(current_date + '.csv', 'w+', newline='')
csv_writer = csv.writer(f)

while True:
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Find face locations and encodings
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    face_names = []

    for face_encoding in face_encodings:
     matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
     name = "Unknown"
     face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
     best_match_index = np.argmin(face_distances)

     if matches[best_match_index]:
          name = known_face_names[best_match_index]

     face_names.append(name)
     current_time = now.strftime("%H-%M-%S")

     if name == "Unknown":
        name = input("Enter the name: ")
        if name in person_info:
            roll_number = person_info[name]["roll_number"]
            gr_number = person_info[name]["gr_number"]
            csv_writer.writerow([name, roll_number, gr_number, current_time])  # Write to CSV

# Check if recognized name is in students list
     if name in students:
         students.remove(name)
         print(students)
         roll_number = person_info[name]["roll_number"]  # Retrieve roll number using recognized name
         gr_number = person_info[name]["gr_number"]  # Retrieve GR number using recognized name
         csv_writer.writerow([name, roll_number, gr_number, current_time])  # Write to CSV

    # Draw rectangles and names on the frame
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the frame
    cv2.imshow("Attendance System", frame)
    
    # Exit loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close the CSV file
video_capture.release()
cv2.destroyAllWindows()
f.close()
