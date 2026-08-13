import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

mpHands = mp.solutions.hands
hands = mpHands.Hands()

draw = mp.solutions.drawing_utils

while True:

    success, img = cap.read()

    if not success:
        print("Failed to read camera")
        break

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:

        for hand in result.multi_hand_landmarks:

            draw.draw_landmarks(
                img,
                hand,
                mpHands.HAND_CONNECTIONS
            )

    cv2.imshow("Hand", img)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()