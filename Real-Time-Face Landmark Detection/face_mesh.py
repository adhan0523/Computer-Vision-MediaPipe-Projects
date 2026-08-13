import cv2
import mediapipe as mp


# ============================================================
# MEDIAPIPE DRAWING SETUP
# ============================================================

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# Custom drawing specification
my_drawing_specs = mp_drawing.DrawingSpec(
    color=(0, 255, 0),
    thickness=1
)


# ============================================================
# CAMERA SETUP
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open camera. Try changing VideoCapture(1) to VideoCapture(0)."
    )


# ============================================================
# MEDIAPIPE FACE MESH
# ============================================================

mp_face_mesh = mp.solutions.face_mesh


with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    # ========================================================
    # MAIN CAMERA LOOP
    # ========================================================

    while cap.isOpened():

        # ----------------------------------------------------
        # READ CAMERA
        # ----------------------------------------------------

        success, image = cap.read()

        if not success:
            print("Could not read camera frame.")
            break


        # ----------------------------------------------------
        # FLIP IMAGE
        # ----------------------------------------------------

        image = cv2.flip(image, 1)


        # ----------------------------------------------------
        # CONVERT BGR TO RGB
        # ----------------------------------------------------

        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )


        # ----------------------------------------------------
        # PROCESS FACE
        # ----------------------------------------------------

        results = face_mesh.process(
            image_rgb
        )


        # ----------------------------------------------------
        # DRAW FACE LANDMARKS
        # ----------------------------------------------------

        if results.multi_face_landmarks:

            for face_landmarks in results.multi_face_landmarks:

                # ------------------------------------------------
                # FACE TESSELATION
                # ------------------------------------------------

                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=(
                        mp_drawing_styles
                        .get_default_face_mesh_tesselation_style()
                    )
                )


                # ------------------------------------------------
                # FACE CONTOURS
                # ------------------------------------------------

                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=my_drawing_specs
                )


                # ------------------------------------------------
                # IRIS LANDMARKS
                # ------------------------------------------------

                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=(
                        mp_drawing_styles
                        .get_default_face_mesh_iris_connections_style()
                    )
                )


        # ----------------------------------------------------
        # SHOW WINDOW
        # ----------------------------------------------------

        cv2.imshow(
            "My video capture",
            image
        )


        # ----------------------------------------------------
        # QUIT
        # ----------------------------------------------------

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()