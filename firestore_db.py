import firebase_admin
from firebase_admin import credentials, firestore
import os
from dotenv import load_dotenv

load_dotenv()

cred = credentials.Certificate(
    os.getenv("FIREBASE_KEY")
)

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

db = firestore.client()


def save_face_data(
    user_id,
    name,
    embedding,
    image_b64,
    filename
):

    db.collection("faces").document(
        user_id
    ).set({

        "name": name,

        "embedding":
            embedding.tolist(),

        "image":
            image_b64,

        "filename":
            filename,

        "created_at":
            firestore.SERVER_TIMESTAMP
    })


def get_all_embeddings():

    docs = db.collection(
        "faces"
    ).stream()

    people = []

    for doc in docs:

        data = doc.to_dict()

        people.append({

            "id":
                doc.id,

            "name":
                data["name"],

            "embedding":
                data["embedding"]
        })

    return people