import cv2
import numpy as np
import tensorflow as tf

# ─────────────────────────────────────────
# Sem tensorflow_hub, sem downloads.
# Usa apenas tf.image (já vem no tensorflow)
# + OpenCV para pré-processamento clássico.
# ─────────────────────────────────────────


# ─────────────────────────────────────────
# ETAPA 1 — Pré-processamento clássico
# ─────────────────────────────────────────

def _classical_enhance(image_bgr: np.ndarray) -> np.ndarray:
    """
    Denoising + CLAHE.
    Melhora imagens escuras, granuladas ou de baixa qualidade.
    """

    denoised = cv2.fastNlMeansDenoisingColored(
        image_bgr, None,
        h=10, hColor=10,
        templateWindowSize=7,
        searchWindowSize=21
    )

    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────
# ETAPA 2 — Super-resolução com tf.image
# Upscale 4× bicúbico + unsharp mask
# Recupera bordas e texturas sem modelo externo
# ─────────────────────────────────────────

def _tf_enhance(image_bgr: np.ndarray) -> np.ndarray:
    """
    1. Converte para tensor float [0,1]
    2. Upscale 4× com bicúbico (tf.image.resize)
    3. Unsharp mask via blur gaussiano — realça bordas
    4. Retorna BGR uint8
    """

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    # float [0, 1]
    tensor = tf.cast(rgb, tf.float32) / 255.0
    tensor = tf.expand_dims(tensor, 0)          # [1, H, W, 3]

    # Upscale 4×
    up = tf.image.resize(
        tensor,
        [h * 4, w * 4],
        method=tf.image.ResizeMethod.BICUBIC,
        antialias=True
    )

    # Unsharp mask — realça detalhes
    blur = tf.nn.avg_pool2d(
        up,
        ksize=5,
        strides=1,
        padding="SAME"
    )
    sharp = tf.clip_by_value(up + 0.6 * (up - blur), 0.0, 1.0)

    # Volta para uint8 BGR
    result = tf.squeeze(sharp).numpy()
    result = (result * 255).astype(np.uint8)

    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


# ─────────────────────────────────────────
# DIAGNÓSTICO — detecta borrão
# ─────────────────────────────────────────

def _is_blurry(image_bgr: np.ndarray, threshold: float = 80.0) -> bool:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold


# ─────────────────────────────────────────
# FUNÇÃO PRINCIPAL
# ─────────────────────────────────────────

def enhance_face(
    face_bgr: np.ndarray,
    force: bool = False,
    blur_threshold: float = 80.0
) -> np.ndarray:
    """
    Melhora rosto recortado antes do FaceNet.

    Parâmetros
    ----------
    face_bgr       : imagem BGR recortada (saída do extract_face)
    force          : força enhancement mesmo em imagens nítidas
    blur_threshold : limiar de variância Laplaciana

    Retorno
    -------
    face BGR 160×160 melhorada
    """

    # 1. Pré-processamento clássico (sempre)
    face = _classical_enhance(face_bgr)

    # 2. Enhancement neural só se necessário
    if force or _is_blurry(face, threshold=blur_threshold):
        face = _tf_enhance(face)

    # 3. Redimensiona para 160×160 (FaceNet)
    face = cv2.resize(face, (160, 160), interpolation=cv2.INTER_LANCZOS4)

    return face