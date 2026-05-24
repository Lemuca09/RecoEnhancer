import cv2
import os
import time
import base64

from face_encoder import (
    get_embedding,
    get_embedding_and_face,
    face_cascade
)

from firestore_db import (
    save_face_data
)

# =========================
# DATASET
# =========================

DATASET_DIR = "dataset"

os.makedirs(
    DATASET_DIR,
    exist_ok=True
)

# =========================
# CAMERA CELULAR
# =========================

def open_cell_camera():

    print("\n1 IP Webcam")
    print("2 DroidCam")
    print("3 URL completa")

    app = input("Escolha: ")

    if app == "1":

        ip = input(
            "IP celular: "
        )

        port = input(
            "Porta [8080]: "
        ) or "8080"

        url = (
            f"http://{ip}:{port}/video"
        )

    elif app == "2":

        ip = input(
            "IP DroidCam: "
        )

        port = input(
            "Porta [4747]: "
        ) or "4747"

        url = (
            f"http://{ip}:{port}/video"
        )

    else:

        url = input(
            "URL completa: "
        )

    cap = cv2.VideoCapture(url)

    if not cap.isOpened():

        print(
            "Erro conexão celular"
        )

        return None

    return cap


# =========================
# SALVAR FACE
# =========================

def save_face(
    frame,
    user_id,
    name
):

    embedding, enhanced_face = get_embedding_and_face(
        frame
    )

    if embedding is None:

        print(
            "Rosto não detectado"
        )

        return False

    timestamp = int(
        time.time()
    )

    filename = (
        f"{name}_{timestamp}.png"
    )

    filepath = os.path.join(
        DATASET_DIR,
        filename
    )

    # SALVAR FACE ENHANCED LOCAL (160×160)
    cv2.imwrite(
        filepath,
        enhanced_face
    )

    # FACE ENHANCED -> JPEG comprimido -> BASE64
    # Firestore tem limite de ~1MB por campo.
    # A face já tem 160×160, JPEG quality=85
    # fica bem abaixo do limite.
    _, buffer = cv2.imencode(
        ".jpg",
        enhanced_face,
        [cv2.IMWRITE_JPEG_QUALITY, 85]
    )

    image_b64 = base64.b64encode(
        buffer
    ).decode(
        "utf-8"
    )

    # FIRESTORE
    save_face_data(
        user_id,
        name,
        embedding,
        image_b64,
        filename
    )

    print(
        "\nFace salva!"
    )

    print(filepath)

    return True


# =========================
# MENU
# =========================

print("\n=== Cadastro Facial ===")
print("1 Webcam PC")
print("2 Camera celular")
print("3 Upload")

mode = input("Escolha: ")

user_id = input(
    "User ID: "
)

name = input(
    "Nome: "
)

# =========================
# UPLOAD
# =========================

if mode == "3":

    path = input(
        "Imagem: "
    )

    frame = cv2.imread(
        path
    )

    if frame is None:

        print(
            "Imagem inválida"
        )

        exit()

    ok = save_face(
        frame,
        user_id,
        name
    )

    if ok:
        print(
            "Upload concluído"
        )

    exit()


# =========================
# CAMERA
# =========================

if mode == "1":

    cap = cv2.VideoCapture(0)

elif mode == "2":

    cap = open_cell_camera()

    if cap is None:
        exit()

else:
    exit()

print(
    "\nMostrando camera..."
)

stable_frames = 0
required_frames = 15

# =========================
# LOOP
# =========================

while True:

    ret, frame = cap.read()

    if not ret:
        continue

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        1.3,
        5
    )

    if len(faces) > 0:

        x, y, w, h = faces[0]

        cv2.rectangle(
            frame,
            (x,y),
            (x+w,y+h),
            (0,255,0),
            2
        )

        stable_frames += 1

        cv2.putText(
            frame,
            f"Detectado {stable_frames}/{required_frames}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

        if stable_frames >= required_frames:

            ok = save_face(
                frame,
                user_id,
                name
            )

            if ok:

                cv2.putText(
                    frame,
                    "SALVO",
                    (20,80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    3
                )

                cv2.imshow(
                    "Cadastro",
                    frame
                )

                cv2.waitKey(2000)

                break

    else:

        stable_frames = 0

        cv2.putText(
            frame,
            "Procurando rosto...",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,0,255),
            2
        )

    cv2.imshow(
        "Cadastro Facial",
        frame
    )

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()