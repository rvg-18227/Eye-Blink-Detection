from scipy.spatial import distance as dist
from imutils import face_utils
import dlib
import cv2
import pandas as pd
import constants


def main():
    df = pd.read_csv("Images/metadata.csv")
    images = df['Image']
    blink = df['Blink']
    prediction = []

    (detector, predictor, l_start, l_end, r_start, r_end) = setup_detector(constants.PREDICTOR_FILE)

    for i in range(len(images)):
        img = cv2.imread(images[i])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)
        val = -1

        for face in faces:
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)
            left_eye = shape[l_start:l_end]
            right_eye = shape[r_start:r_end]
            left_eye_ar = eye_aspect_ratio(left_eye)
            right_eye_ar = eye_aspect_ratio(right_eye)
            ear = (left_eye_ar + right_eye_ar) / 2.0

            if ear < constants.EAR_THRESH:
                val = 1
            else:
                val = 0

        prediction.append(val)

    result_dict = {"Image": images, "Blink": blink, "Prediction": prediction}
    result_df = pd.DataFrame(result_dict)
    result_df.to_csv("results.csv")


def eye_aspect_ratio(eye):
    # Computing the distance between the two sets of vertical eye landmarks
    a = dist.euclidean(eye[1], eye[5])
    b = dist.euclidean(eye[2], eye[4])

    # Computing the distance between the set of horizontal eye landmarks
    c = dist.euclidean(eye[0], eye[3])

    # Computing the EAR
    ear = (a + b) / (2.0 * c)

    return ear


def setup_detector(predictor_file):
    # Initializing dlib HOG based face detector
    print("--- Loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()

    # Creating facial landmark predictor
    predictor = dlib.shape_predictor(predictor_file)

    # Obtaining the facial landmarks for the left and right eye
    (l_start, l_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (r_start, r_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    return detector, predictor, l_start, l_end, r_start, r_end


if __name__ == "__main__":
    main()
