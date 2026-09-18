"""One-click execution script for Experiment 14: Multi-Modal CLIP Contrastive Learning."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from .dataset import (
    ALL_SHAPES,
    COLORS,
    KNOWN_SHAPES,
    NOVEL_SHAPES,
    COMPOSITIONAL_HOLDOUT,
    draw_geometric_shape,
)
from .train import train_clip_model
from .losses import perform_cross_modal_retrieval
from .visualize import (
    plot_architecture_overview,
    plot_cross_modal_retrieval,
    plot_joint_embedding_space,
    plot_similarity_matrix_heatmaps,
    plot_training_dynamics,
    plot_zero_shot_showcase,
)


def run_experiment():
    output_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 75)
    print("  EXPERIMENT 14: CONTRASTIVE LANGUAGE-IMAGE PRE-TRAINING (CLIP)")
    print("  Showcasing Dual-Encoders, InfoNCE Loss, Compositional & Novel Shape Zero-Shot")
    print("=" * 75 + "\n")

    # 1. Architecture Overview Diagram
    arch_path = os.path.join(output_dir, "clip_architecture_and_infonce.png")
    plot_architecture_overview(save_path=arch_path)
    print(f"[1/6] Generated architecture diagram -> {arch_path}")

    # 2. Train Model
    clip_model, vectorizer, history, eval_results = train_clip_model(
        epochs=16,
        batch_size=64,
        learning_rate=1e-3,
        embedding_dim=64,
        samples_per_combo=120,
        seed=42,
    )

    # 3. Training Dynamics Plot
    dyn_path = os.path.join(output_dir, "clip_training_dynamics.png")
    plot_training_dynamics(history, save_path=dyn_path)
    print(f"[2/6] Generated training dynamics -> {dyn_path}")

    # 4. Similarity Matrix Heatmap Plot
    sim_path = os.path.join(output_dir, "clip_similarity_matrix_heatmap.png")
    plot_similarity_matrix_heatmaps(
        eval_results["sim_before"],
        eval_results["sim_after"],
        eval_results["sample_txts"],
        save_path=sim_path,
    )
    print(f"[3/6] Generated similarity heatmap -> {sim_path}")

    # 5. Zero-Shot Qualitative Showcase (Seen, Compositional, Novel Star, Novel Diamond)
    all_candidate_classes = [f"{c}_{s}" for s in ALL_SHAPES for c in COLORS.keys()]
    candidate_prompts = [f"a photo of a {c.replace('_', ' ')}" for c in all_candidate_classes]
    tokenized_all_prompts = vectorizer(tf.constant(candidate_prompts))
    all_text_embeddings = clip_model.encode_texts(tokenized_all_prompts).numpy()

    def get_zs_probabilities(img_arr):
        img_emb = clip_model.encode_images(tf.constant(np.expand_dims(img_arr, 0))).numpy()
        cos_sim = np.matmul(img_emb, all_text_embeddings.T)[0]
        # Softmax with temperature scale
        temp = float(tf.math.exp(clip_model.logit_scale).numpy())
        exp_sim = np.exp(cos_sim * temp)
        probs = exp_sim / np.sum(exp_sim)
        return probs

    # Select representative test cases
    test_cases = []

    # Case 1: Seen combination
    seen_img = draw_geometric_shape("circle", COLORS["red"], jitter=False)
    seen_probs = get_zs_probabilities(seen_img)
    top_indices = np.argsort(-seen_probs)[:5]
    test_cases.append({
        "image": seen_img,
        "category": "Seen In-Distribution",
        "true_label": "red circle",
        "candidate_prompts": [candidate_prompts[i] for i in top_indices],
        "probabilities": seen_probs[top_indices],
    })

    # Case 2: Compositional Zero-Shot (Yellow Square)
    comp_img1 = draw_geometric_shape("square", COLORS["yellow"], jitter=False)
    comp_probs1 = get_zs_probabilities(comp_img1)
    top_indices = np.argsort(-comp_probs1)[:5]
    test_cases.append({
        "image": comp_img1,
        "category": "Compositional Zero-Shot (Unseen Pair)",
        "true_label": "yellow square",
        "candidate_prompts": [candidate_prompts[i] for i in top_indices],
        "probabilities": comp_probs1[top_indices],
    })

    # Case 3: Compositional Zero-Shot (Cyan Triangle)
    comp_img2 = draw_geometric_shape("triangle", COLORS["cyan"], jitter=False)
    comp_probs2 = get_zs_probabilities(comp_img2)
    top_indices = np.argsort(-comp_probs2)[:5]
    test_cases.append({
        "image": comp_img2,
        "category": "Compositional Zero-Shot (Unseen Pair)",
        "true_label": "cyan triangle",
        "candidate_prompts": [candidate_prompts[i] for i in top_indices],
        "probabilities": comp_probs2[top_indices],
    })

    # Case 4: Novel Geometric Figure (Yellow Star)
    star_img = draw_geometric_shape("star", COLORS["yellow"], jitter=False)
    star_probs = get_zs_probabilities(star_img)
    top_indices = np.argsort(-star_probs)[:5]
    test_cases.append({
        "image": star_img,
        "category": "Novel Geometric Figure (Star)",
        "true_label": "yellow star",
        "candidate_prompts": [candidate_prompts[i] for i in top_indices],
        "probabilities": star_probs[top_indices],
    })

    # Case 5: Novel Geometric Figure (Purple Diamond)
    diamond_img = draw_geometric_shape("diamond", COLORS["purple"], jitter=False)
    diamond_probs = get_zs_probabilities(diamond_img)
    top_indices = np.argsort(-diamond_probs)[:5]
    test_cases.append({
        "image": diamond_img,
        "category": "Novel Geometric Figure (Diamond)",
        "true_label": "purple diamond",
        "candidate_prompts": [candidate_prompts[i] for i in top_indices],
        "probabilities": diamond_probs[top_indices],
    })

    zs_path = os.path.join(output_dir, "clip_zero_shot_classification.png")
    plot_zero_shot_showcase(test_cases, save_path=zs_path)
    print(f"[4/6] Generated zero-shot classification gallery -> {zs_path}")

    # 6. Cross-Modal Text-to-Image Retrieval
    query_prompts = [
        "a photo of a blue square",
        "a green triangle",
        "a photo of a yellow circle",
        "a purple hexagon",
    ]
    val_images = eval_results["val_data"]["images"]
    val_texts = eval_results["val_data"]["texts"]
    val_labels = eval_results["val_data"]["labels"]

    tokenized_queries = vectorizer(tf.constant(query_prompts))
    query_txt_emb = clip_model.encode_texts(tokenized_queries).numpy()
    val_img_emb = clip_model.encode_images(tf.constant(val_images)).numpy()

    retrieval_images = []
    retrieval_scores = []
    for q_emb in query_txt_emb:
        sims = np.matmul(val_img_emb, q_emb)
        top5_idx = np.argsort(-sims)[:5]
        retrieval_images.append([val_images[i] for i in top5_idx])
        retrieval_scores.append([sims[i] for i in top5_idx])

    retrieval_path = os.path.join(output_dir, "clip_cross_modal_retrieval.png")
    plot_cross_modal_retrieval(query_prompts, retrieval_images, retrieval_scores, save_path=retrieval_path)
    print(f"[5/6] Generated cross-modal retrieval showcase -> {retrieval_path}")

    # 7. Joint Multimodal Embedding Space (PCA 2D)
    sample_vis_classes = ["red_circle", "blue_square", "green_triangle", "yellow_square"]
    vis_imgs, vis_txts, vis_lbls = [], [], []

    for c_lbl in sample_vis_classes:
        # 15 samples per class from val or comp
        if "yellow_square" in c_lbl:
            src = eval_results["comp_data"]
        else:
            src = eval_results["val_data"]
        mask = src["labels"] == c_lbl
        selected_imgs = src["images"][mask][:12]
        selected_txts = src["texts"][mask][:12]
        for img, txt in zip(selected_imgs, selected_txts):
            vis_imgs.append(img)
            vis_txts.append(txt)
            vis_lbls.append(c_lbl.replace("_", " "))

    vis_img_emb = clip_model.encode_images(tf.constant(np.array(vis_imgs))).numpy()
    vis_txt_emb = clip_model.encode_texts(vectorizer(tf.constant(vis_txts))).numpy()

    emb_path = os.path.join(output_dir, "clip_joint_embedding_space.png")
    plot_joint_embedding_space(vis_img_emb, vis_txt_emb, vis_lbls, save_path=emb_path)
    print(f"[6/6] Generated shared metric space plot -> {emb_path}")

    print("\n" + "=" * 75)
    print("  ALL CLIP EXPERIMENTS & VISUALIZATIONS GENERATED SUCCESSFULLY!")
    print(f"  Figures saved in: {output_dir}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    run_experiment()
