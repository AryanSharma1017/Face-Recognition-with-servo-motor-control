Face Recognition:


import cv2
import face_recognition
import os
import RPi.GPIO as GPIO
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Create a dictionary to store known face encodings and names
known_face_encodings = {}
known_face_names = {}

# Path to your dataset directory
dataset_dir = "/home/aryan/dataset"

# Path to a directory where unknown person images will be saved
unknown_dir = "/home/aryan/unknown"

# Define GPIO pins for servo motor
servo_pin = 18  # Change this to the actual GPIO pin connected to your servo
GPIO.setmode(GPIO.BCM)
GPIO.setup(servo_pin, GPIO.OUT)

# Function to move the servo to a specified angle
def move_servo(angle):
    duty_cycle = angle / 18.0 + 2.5
    GPIO.output(servo_pin, True)
    pwm.ChangeDutyCycle(duty_cycle)

# Initialize PWM for servo control
pwm = GPIO.PWM(servo_pin, 50)
pwm.start(0)

# Email configuration
smtp_server = "smtp.gmail.com"  # Replace with your SMTP server address
smtp_port = 587  # Replace with your SMTP server's port (usually 587 for TLS)
smtp_username = ""  # Replace with your email address
smtp_password = ""  # Replace with your email password
sender_email = ""  # Replace with your email address
receiver_email = ""  # Replace with the recipient's email address

# Function to send an email with the captured image attached
def send_email_with_image(image_filename):
    subject = "Unknown Person Detected"
    message = "An unknown person has been detected. See attached image."

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject

    msg.attach(MIMEText(message, 'plain'))

    # Attach the captured image
    with open(image_filename, 'rb') as image_file:
        image_data = image_file.read()
        image = MIMEImage(image_data, name=os.path.basename(image_filename))
        msg.attach(image)

    # Connect to the SMTP server and send the email
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_username, smtp_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        print("Email sent successfully")
    except Exception as e:
        print(f"Error sending email: {str(e)}")

# Loop through each subdirectory (each person) in the dataset directory
for person_name in os.listdir(dataset_dir):
    person_dir = os.path.join(dataset_dir, person_name)

    if os.path.isdir(person_dir):
        known_face_encodings[person_name] = []
        known_face_names[person_name] = []

        # Loop through each image in the person's subdirectory
        for filename in os.listdir(person_dir):
            image_path = os.path.join(person_dir, filename)
            image = face_recognition.load_image_file(image_path)

            # Check if faces are detected in the image
            face_locations = face_recognition.face_locations(image)

            if len(face_locations) > 0:
                # Assuming only one face is in the image, you can modify this if needed
                face_encoding = face_recognition.face_encodings(image, face_locations)[0]

                # Add the face encoding and name to the dictionaries
                known_face_encodings[person_name].append(face_encoding)
                known_face_names[person_name].append(person_name)
            else:
                print(f"No faces found in {image_path}")

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Resize the frame to speed up face recognition (optional)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the frame from BGR to RGB (required for face_recognition)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Find all face locations and encodings in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each detected face
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "Unknown"  # Default name if no match is found

        # Loop through known people in your dataset
        for person_name in known_face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings[person_name], face_encoding)

            # If a match is found, use the name of the known person
            if True in matches:
                name = person_name
                break

        # Draw a rectangle around the face and label it with the person's name
        cv2.rectangle(frame, (left * 4, top * 4), (right * 4, bottom * 4), (0, 255, 0), 2)
        cv2.putText(frame, name, (left * 4, bottom * 4 + 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 1)

        # Control the servo motor based on face detection
        if name != "Unknown":
            move_servo(180)  # Move the servo to 180 degrees
        else:
            move_servo(0)    # Move the servo to 0 degrees for unknown person
            
            # Save the frame containing the unknown person's face
            unknown_image_filename = os.path.join(unknown_dir, f"unknown_{int(time.time())}.jpg")
            cv2.imwrite(unknown_image_filename, frame)

            # Send an email with the captured image
            send_email_with_image(unknown_image_filename)

    # Display the frame with detected faces and names
    cv2.imshow('Face Recognition', frame)

    # Break the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam, stop the PWM, and close all OpenCV windows
cap.release()
pwm.stop()
GPIO.cleanup()
cv2.destroyAllWindows()
