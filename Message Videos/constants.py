# The eye aspect ratio (EAR) threshold to consider the eyes being closed
EAR_THRESH = 0.26
# The minimum number of frames that the eyes need to be closed to indicate a short blink (dot)
EAR_FRAMES_DOT = 4
# The minimum number of frames that the eyes need to be closed to indicate a long blink (dash)
EAR_FRAMES_DASH = 12
# The minimum number of frames the eye must be open to indicate a change of character
CHAR_PAUSE_FRAMES = 25
# The minimum number of frames the eye must be open to indicate a change of word (along with CHAR_PAUSE_FRAMES)
WORD_PAUSE_FRAMES = 35
# The minimum number of frames that the eyes need to be closed to exit the program
BREAK_LOOP_FRAMES = 60
# Reference to the predictor file to predict facial landmarks
PREDICTOR_FILE = "shape_predictor_68_face_landmarks.dat"
