from keras_facenet import FaceNet
import cv2
import numpy as np

from face_enhancer import enhance_face

embedder = FaceNet()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


def extract_face(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]

    face = image[y:y+h, x:x+w]
    face = cv2.resize(face, (160, 160))

    return face


def get_embedding_and_face(image, enhance=True):
    """
    Igual ao get_embedding, mas também retorna a face
    processada — útil para salvar a imagem melhorada.

    Retorno
    -------
    (embedding np.ndarray 512, face BGR 160×160) ou (None, None)
    """

    face = extract_face(image)

    if face is None:
        return None, None

    if enhance:
        face = enhance_face(face)

    embedding = embedder.embeddings(
        np.expand_dims(face, axis=0)
    )

    return embedding[0], face


def get_embedding(image, enhance=True):
    """
    Extrai embedding facial de uma imagem.

    Parâmetros
    ----------
    image   : frame BGR (câmera ou arquivo)
    enhance : se True, roda o pipeline de melhoria
              antes de gerar o embedding.
              Desative para benchmarking ou imagens
              já nítidas.

    Retorno
    -------
    np.ndarray de 512 floats ou None se sem rosto.
    """

    face = extract_face(image)

    if face is None:
        return None

    # ── ENHANCEMENT ─────────────────────────
    # Roda antes do FaceNet para que o modelo
    # receba a versão mais nítida possível do
    # rosto, melhorando a qualidade do embedding
    # e consequentemente o cosine similarity.
    if enhance:
        face = enhance_face(face)
    # ────────────────────────────────────────

    face = np.expand_dims(face, axis=0)

    embedding = embedder.embeddings(face)

    return embedding[0]