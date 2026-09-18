"""Loss functions, metrics, and evaluation routines for CLIP contrastive learning."""

from typing import Dict, List, Tuple
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def compute_similarity_matrix(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    logit_scale: float = 1.0,
) -> np.ndarray:
    """Compute pairwise cosine similarity matrix scaled by logit scale."""
    # Dot product between unit normalized embeddings
    sim = np.matmul(image_embeddings, text_embeddings.T) * logit_scale
    return sim


def evaluate_zero_shot_classification(
    model: keras.Model,
    vectorizer: layers.TextVectorization,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    candidate_classes: List[str],
    prompt_template: str = "a photo of a {}",
) -> Dict[str, float]:
    """Perform zero-shot classification by ranking text prompts against test images."""
    # 1. Generate text prompts for all candidate classes
    candidate_prompts = [prompt_template.format(c.replace("_", " ")) for c in candidate_classes]
    tokenized_prompts = vectorizer(tf.constant(candidate_prompts))

    # 2. Encode all candidate text prompts
    text_embeddings = model.encode_texts(tokenized_prompts).numpy()

    # 3. Encode test images
    img_embeddings = model.encode_images(tf.constant(test_images)).numpy()

    # 4. Compute cosine similarity: [num_images, num_candidates]
    similarity = np.matmul(img_embeddings, text_embeddings.T)

    # 5. Predictions: argmax over candidate classes
    pred_indices = np.argmax(similarity, axis=-1)
    predicted_classes = [candidate_classes[i] for i in pred_indices]

    # Compute accuracy
    correct = np.array(predicted_classes) == test_labels
    top1_acc = np.mean(correct)

    return {
        "top1_acc": float(top1_acc),
        "predicted_classes": predicted_classes,
        "similarity_matrix": similarity,
        "candidate_prompts": candidate_prompts,
    }


def perform_cross_modal_retrieval(
    model: keras.Model,
    vectorizer: layers.TextVectorization,
    images: np.ndarray,
    texts: np.ndarray,
    top_k: int = 5,
) -> Dict[str, np.ndarray]:
    """Compute top-k nearest neighbors for image-to-text and text-to-image retrieval."""
    tokenized_texts = vectorizer(tf.constant(texts))
    img_emb = model.encode_images(tf.constant(images)).numpy()
    txt_emb = model.encode_texts(tokenized_texts).numpy()

    # Cosine similarity matrix: [N, N]
    sim = np.matmul(img_emb, txt_emb.T)

    # Top-K text matches for each image
    i2t_topk = np.argsort(-sim, axis=1)[:, :top_k]

    # Top-K image matches for each text
    t2i_topk = np.argsort(-sim.T, axis=1)[:, :top_k]

    return {
        "i2t_topk": i2t_topk,
        "t2i_topk": t2i_topk,
        "similarity": sim,
        "img_emb": img_emb,
        "txt_emb": txt_emb,
    }
