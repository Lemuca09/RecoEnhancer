import cv2
import os
import requests
import numpy as np

from dotenv import load_dotenv

from face_encoder import get_embedding
from firestore_db import get_all_embeddings
from utils import cosine_sim

load_dotenv()

THRESHOLD = float(
    os.getenv(
        "THRESHOLD",
        0.65
    )
)

db_faces = get_all_embeddings()

print("1 Webcam PC")
print("2 Camera celular")
print("3 Upload")

mode = input("Escolha: ")


def identify(emb):

    label = "Unknown"
    best = 0

    for p in db_faces:

        score = cosine_sim(
            emb,
            p["embedding"]
        )

        if (
            score > best
            and score >= THRESHOLD
        ):
            best = score

            label = (
                f'{p["name"]} '
                f'({score:.2f})'
            )

    return label


# =====================
# WEBCAM PC
# =====================

if mode == "1":

    cap = cv2.VideoCapture(0)

    while True:

        ret, frame = cap.read()

        emb = get_embedding(frame)

        label = "Unknown"

        if emb is not None:
            label = identify(emb)

        cv2.putText(
            frame,
            label,
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        cv2.imshow(
            "Recognition",
            frame
        )

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# =====================
# CAMERA CELULAR
# =====================

elif mode == "2":

    ip = input(
        "URL IP Webcam: "
    )

    url = f"{ip}/shot.jpg"

    while True:

        img_resp = requests.get(url)

        img_arr = np.array(
            bytearray(img_resp.content),
            dtype=np.uint8
        )

        frame = cv2.imdecode(
            img_arr,
            -1
        )

        emb = get_embedding(frame)

        label = "Unknown"

        if emb is not None:
            label = identify(emb)

        cv2.putText(
            frame,
            label,
            (30,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        cv2.imshow(
            "Cellphone Recognition",
            frame
        )

        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyAllWindows()


# =====================
# UPLOAD
# =====================

elif mode == "3":

    path = input(
        "Imagem: "
    )

    image = cv2.imread(path)

    emb = get_embedding(image)

    if emb is None:
        print("Rosto não detectado")
        exit()

    print(
        "Resultado:",
        identify(emb)
    )