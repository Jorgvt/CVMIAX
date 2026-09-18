"""Training and evaluation pipeline for CLIP multimodal contrastive learning."""

import os
from typing import Dict, Tuple
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .dataset import (
    ALL_SHAPES,
    COLORS,
    COMPOSITIONAL_HOLDOUT,
    KNOWN_SHAPES,
    NOVEL_SHAPES,
    build_text_vectorizer,
    create_multimodal_dataset,
    create_tf_dataset,
)
from .losses import evaluate_zero_shot_classification, perform_cross_modal_retrieval
from .models import CLIPDualEncoder, build_text_encoder, build_vision_encoder


def train_clip_model(
    epochs: int = 18,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    embedding_dim: int = 64,
    samples_per_combo: int = 120,
    seed: int = 42,
) -> Tuple[CLIPDualEncoder, layers.TextVectorization, Dict, Dict]:
    """Train CLIP model and evaluate zero-shot & cross-modal retrieval performance."""
    print("=" * 70)
    print("STEP 1: Generating Multimodal Dataset (Images + Natural Language)")
    print("=" * 70)

    train_data, val_data, comp_data, novel_data = create_multimodal_dataset(
        num_samples_per_combo=samples_per_combo,
        seed=seed,
    )
    print(f"Train samples: {len(train_data['images'])}")
    print(f"Val samples:   {len(val_data['images'])}")
    print(f"Compositional zero-shot test samples: {len(comp_data['images'])}")
    print(f"Novel shape test samples:            {len(novel_data['images'])}")

    # Build Text Vectorizer
    vectorizer = build_text_vectorizer(train_data["texts"])
    vocab_size = vectorizer.vocabulary_size()
    seq_length = 8

    # Create tf.data.Dataset
    train_ds = create_tf_dataset(train_data, vectorizer, batch_size=batch_size, shuffle=True)
    val_ds = create_tf_dataset(val_data, vectorizer, batch_size=batch_size, shuffle=False)

    print("\n" + "=" * 70)
    print("STEP 2: Building Dual Encoders & CLIP Architecture")
    print("=" * 70)

    vision_enc = build_vision_encoder(input_shape=(64, 64, 3), embedding_dim=embedding_dim)
    text_enc = build_text_encoder(vocab_size=vocab_size, seq_length=seq_length, embedding_dim=embedding_dim)

    clip_model = CLIPDualEncoder(
        vision_encoder=vision_enc,
        text_encoder=text_enc,
        init_temperature=0.07,
    )

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    clip_model.compile(optimizer=optimizer)

    # Record initial similarity matrix before training
    sample_imgs = val_data["images"][:32]
    sample_txts = val_data["texts"][:32]
    sample_tokens = vectorizer(tf.constant(sample_txts))
    img_emb_before = clip_model.encode_images(tf.constant(sample_imgs)).numpy()
    txt_emb_before = clip_model.encode_texts(sample_tokens).numpy()
    sim_before = np.matmul(img_emb_before, txt_emb_before.T)

    print("\n" + "=" * 70)
    print("STEP 3: Training with Symmetric InfoNCE Contrastive Loss")
    print("=" * 70)

    history = {
        "loss": [],
        "i2t_acc": [],
        "t2i_acc": [],
        "val_loss": [],
        "val_i2t_acc": [],
        "val_t2i_acc": [],
        "temperature": [],
    }

    for epoch in range(1, epochs + 1):
        # Training epoch
        epoch_losses, epoch_i2t, epoch_t2i = [], [], []
        for batch_imgs, batch_txts in train_ds:
            metrics = clip_model.train_step((batch_imgs, batch_txts))
            epoch_losses.append(float(metrics["loss"]))
            epoch_i2t.append(float(metrics["i2t_acc"]))
            epoch_t2i.append(float(metrics["t2i_acc"]))

        # Validation epoch
        val_losses, val_i2t, val_t2i = [], [], []
        for v_imgs, v_txts in val_ds:
            v_metrics = clip_model.test_step((v_imgs, v_txts))
            val_losses.append(float(v_metrics["loss"]))
            val_i2t.append(float(v_metrics["i2t_acc"]))
            val_t2i.append(float(v_metrics["t2i_acc"]))

        cur_temp = float(tf.math.exp(clip_model.logit_scale).numpy())
        history["loss"].append(np.mean(epoch_losses))
        history["i2t_acc"].append(np.mean(epoch_i2t))
        history["t2i_acc"].append(np.mean(epoch_t2i))
        history["val_loss"].append(np.mean(val_losses))
        history["val_i2t_acc"].append(np.mean(val_i2t))
        history["val_t2i_acc"].append(np.mean(val_t2i))
        history["temperature"].append(cur_temp)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} - Loss: {history['loss'][-1]:.4f} "
            f"| I2T Acc: {history['i2t_acc'][-1]*100:.1f}% | T2I Acc: {history['t2i_acc'][-1]*100:.1f}% "
            f"| Val Loss: {history['val_loss'][-1]:.4f} | Temp Scale: {cur_temp:.2f}"
        )

    # Post-training similarity matrix
    img_emb_after = clip_model.encode_images(tf.constant(sample_imgs)).numpy()
    txt_emb_after = clip_model.encode_texts(sample_tokens).numpy()
    sim_after = np.matmul(img_emb_after, txt_emb_after.T)

    print("\n" + "=" * 70)
    print("STEP 4: Quantitative Zero-Shot Evaluation")
    print("=" * 70)

    # Candidate classes (all known + novel combinations)
    all_candidate_classes = [f"{c}_{s}" for s in ALL_SHAPES for c in COLORS.keys()]
    known_candidate_classes = [f"{c}_{s}" for s in KNOWN_SHAPES for c in COLORS.keys()]

    # 1. In-distribution seen validation accuracy
    val_zs = evaluate_zero_shot_classification(
        clip_model, vectorizer, val_data["images"], val_data["labels"], known_candidate_classes
    )
    # 2. Compositional zero-shot accuracy (withheld color-shape pairs)
    comp_zs = evaluate_zero_shot_classification(
        clip_model, vectorizer, comp_data["images"], comp_data["labels"], known_candidate_classes
    )
    # 3. Novel Shape test accuracy (completely novel shapes: stars & diamonds)
    novel_zs = evaluate_zero_shot_classification(
        clip_model, vectorizer, novel_data["images"], novel_data["labels"], all_candidate_classes
    )

    print(f"Zero-Shot Accuracy [Seen In-Distribution]:       {val_zs['top1_acc']*100:.2f}%")
    print(f"Zero-Shot Accuracy [Compositional Hold-Out]:     {comp_zs['top1_acc']*100:.2f}%")
    print(f"Zero-Shot Accuracy [Novel Shapes (Star/Diamond)]: {novel_zs['top1_acc']*100:.2f}%")

    eval_results = {
        "val_zs": val_zs,
        "comp_zs": comp_zs,
        "novel_zs": novel_zs,
        "sim_before": sim_before,
        "sim_after": sim_after,
        "sample_txts": sample_txts,
        "train_data": train_data,
        "val_data": val_data,
        "comp_data": comp_data,
        "novel_data": novel_data,
    }

    return clip_model, vectorizer, history, eval_results
