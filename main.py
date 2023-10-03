from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import dlib
import cv2
import keyboard
import decode
import constants


def main():
    (vs, detector, predictor, l_start, l_end, r_start, r_end) = setup_detector_video(constants.PREDICTOR_FILE)
    total_morse = loop_camera(vs, detector, predictor, l_start, l_end, r_start, r_end)
    cleanup(vs)
    print_results(total_morse)


def eye_aspect_ratio(eye):
    # Computing the distance between the two sets of vertical eye landmarks
    a = dist.euclidean(eye[1], eye[5])
    b = dist.euclidean(eye[2], eye[4])

    # Computing the distance between the set of horizontal eye landmarks
    c = dist.euclidean(eye[0], eye[3])

    # Computing the EAR
    ear = (a + b) / (2.0 * c)

    return ear


def setup_detector_video(predictor_file):
    # Initializing dlib HOG based face detector
    print("--- Loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()

    # Creating facial landmark predictor
    predictor = dlib.shape_predictor(predictor_file)

    # Obtaining the facial landmarks for the left and right eye
    (l_start, l_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (r_start, r_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # Starting video stream thread
    print("--- Starting video stream thread...")
    print("--- Type ']' or close eyes for {} frames to exit.".format(constants.BREAK_LOOP_FRAMES))
    vs = VideoStream(src=0).start()

    return vs, detector, predictor, l_start, l_end, r_start, r_end


def loop_camera(vs, detector, predictor, l_start, l_end, r_start, r_end):
    # Initializing the frame counter and total blinks
    counter = 0
    break_counter = 0
    eyes_open_counter = 0
    eyes_closed = False
    word_pause = False
    paused = False

    morse_msg = ""
    morse_word = ""
    morse_char = ""

    # Looping over the frames of the video stream
    while True:
        # Read the frame, resize it and convert to grayscale
        frame = vs.read()
        frame = imutils.resize(frame, width=850)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces in the grayscale frame
        faces = detector(gray, 0)

        # Looping over detected faces
        for face in faces:
            # Determine the facial landmarks  and convert the coordinates to an array
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)
            # Extract the left and right eye coordinates and use it to compute the EAR for both eyes
            left_eye = shape[l_start:l_end]
            right_eye = shape[r_start:r_end]
            left_eye_ar = eye_aspect_ratio(left_eye)
            right_eye_ar = eye_aspect_ratio(right_eye)
            # Take the mean of both EAR values
            ear = (left_eye_ar + right_eye_ar) / 2.0

            # Compute the convex hull for both eyes and visualize each of the eyes
            left_eye_hull = cv2.convexHull(left_eye)
            right_eye_hull = cv2.convexHull(right_eye)
            cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)

            # Check to see if the EAR is below the threshold (accordingly we increment the blink frame counter)
            if ear < constants.EAR_THRESH:
                counter += 1
                break_counter += 1
                if counter >= constants.EAR_FRAMES_DOT:
                    eyes_closed = True
                # Reset morse that appears on screen if it had just been "/"
                if not paused:
                    morse_char = ""
                # Eyes closed for long enough to exit the program
                if break_counter >= constants.BREAK_LOOP_FRAMES:
                    break
            # If EAR is not below the threshold
            else:
                # Eyes weren't closed for too long
                if break_counter < constants.BREAK_LOOP_FRAMES:
                    break_counter = 0
                eyes_open_counter += 1
                # Dash detected as eyes closed for a longer duration
                if counter >= constants.EAR_FRAMES_DASH:
                    morse_word += "-"
                    morse_msg += "-"
                    morse_char += "-"
                    # reset the eye frame counter
                    counter = 0
                    eyes_closed = False
                    paused = True
                    eyes_open_counter = 0
                # Dot detected as eyes closed for a shorter duration
                elif eyes_closed:
                    morse_word += "."
                    morse_msg += "."
                    morse_char += "."
                    counter = 1
                    eyes_closed = False
                    paused = True
                    eyes_open_counter = 0
                # Only add spaces between characters if eyes have been open for more than CHAR_PAUSE_FRAMES frames
                elif paused and (eyes_open_counter >= constants.CHAR_PAUSE_FRAMES):
                    morse_word += "/"
                    morse_msg += "/"
                    morse_char = "/"
                    paused = False
                    word_pause = True
                    eyes_closed = False
                    eyes_open_counter = 0
                    keyboard.write(decode.from_morse(morse_word))
                    morse_word = ""
                # Add space between words if eyes have been open for more than WORD_PAUSE_FRAMES frames
                # (This is in addition to CHAR_PAUSE_FRAMES)
                elif (word_pause and eyes_open_counter >=
                      constants.WORD_PAUSE_FRAMES):
                    # "/" already in the string from character pause, "¦" is converted to a space " ".
                    morse_msg += "¦/"
                    morse_char = ""
                    word_pause = False
                    eyes_closed = False
                    eyes_open_counter = 0
                    keyboard.write(decode.from_morse("¦/"))

            # Draw the computed EAR for the frame and display morse code
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "{}".format(morse_char), (100, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

            # Print the recent morse code to the console
            print("\033[K", "morse_word: {}".format(morse_word), end="\r")

        # Show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # If the `]` key was pressed, break from the loop
        if key == ord("]") or (break_counter >= constants.BREAK_LOOP_FRAMES):
            keyboard.write(decode.from_morse(morse_word))
            break

    return morse_msg


def cleanup(vs):
    cv2.destroyAllWindows()
    vs.stop()


def print_results(total_morse):
    print("Morse Code:", total_morse.replace("¦", " "))
    print("Translated:", decode.from_morse(total_morse))


if __name__ == "__main__":
    main()
