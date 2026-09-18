"""
Master Execution Script for Experiment 09: Simple Two-Stage Object Detection (R-CNN).

Executes end-to-end:
1. Synthetic dataset creation with multi-shape canvases.
2. Proposal generation, IoU matching, and balanced crop dataset extraction.
3. CNN training (classification + bbox regression heads) with checkpoint caching.
4. Full inference, NMS, and mAP evaluation.
5. Generating and saving representative figures featuring multi-shape scenes.
"""

import argparse
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Ensure module imports work smoothly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from dataset import generate_dataset, generate_single_sample
from evaluate import detect_objects_rcnn, evaluate_dataset_map
from proposals import (
    extract_and_warp_crops,
    generate_sliding_window_proposals,
    match_proposals_to_ground_truth,
)
from train import (
    load_or_train_rcnn_detector,
    prepare_rcnn_training_data,
)
from visualize import (
    visualize_complete_pipeline,
    visualize_crops_grid,
    visualize_dataset_samples,
    visualize_proposals_and_iou,
)


def run_experiment(force_retrain: bool = False):
    print("=" * 70)
    print(" EXPERIMENT 09: TWO-STAGE OBJECT DETECTION (R-CNN FROM SCRATCH)")
    print("=" * 70)

    checkpoint_dir = os.path.join(CURRENT_DIR, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "rcnn_detector.weights.h5")

    # 1. Dataset Generation
    print("\n[Step 1/5] Generating Synthetic Geometric Multi-Shape Dataset...")
    train_images, train_annos = generate_dataset(num_samples=250, img_size=128, min_objects=1, max_objects=3, seed=42)
    val_images, val_annos = generate_dataset(num_samples=60, img_size=128, min_objects=1, max_objects=3, seed=101)
    test_images, test_annos = generate_dataset(num_samples=40, img_size=128, min_objects=2, max_objects=3, seed=202)

    print(f"Generated: Train={len(train_images)}, Val={len(val_images)}, Test={len(test_images)} images.")

    # Save dataset samples visualization
    samples_fig_path = os.path.join(CURRENT_DIR, "rcnn_01_dataset_samples.png")
    visualize_dataset_samples(train_images, train_annos, num_samples=6, save_path=samples_fig_path)
    plt.close()
    print(f"Saved dataset visualization to {samples_fig_path}")

    # 2. Proposal Extraction & IoU Matching Visualization on a Multi-Shape Image
    print("\n[Step 2/5] Demonstrating Stage 1 Region Proposals & IoU Matching on Multi-Shape Image...")
    # Find a sample with 3 objects
    multi_obj_idx = 0
    for i, anno in enumerate(train_annos):
        if len(anno["boxes"]) >= 3:
            multi_obj_idx = i
            break

    sample_img = train_images[multi_obj_idx]
    sample_boxes = train_annos[multi_obj_idx]["boxes"]
    sample_classes = train_annos[multi_obj_idx]["classes"]

    proposals = generate_sliding_window_proposals(img_size=128)
    matched_labels, target_deltas, max_ious = match_proposals_to_ground_truth(
        proposals, sample_boxes, sample_classes, pos_thresh=0.5, neg_thresh=0.2
    )

    iou_fig_path = os.path.join(CURRENT_DIR, "rcnn_02_proposals_and_iou.png")
    visualize_proposals_and_iou(
        sample_img, sample_boxes, sample_classes, proposals, matched_labels, max_ious, save_path=iou_fig_path
    )
    plt.close()
    print(f"Saved proposals & IoU visualization to {iou_fig_path}")

    # Extract crops visualization
    sample_crops = extract_and_warp_crops(sample_img, proposals[matched_labels >= 0])
    sample_crop_labels = matched_labels[matched_labels >= 0]
    crops_fig_path = os.path.join(CURRENT_DIR, "rcnn_03_warped_crops_grid.png")
    visualize_crops_grid(sample_crops, sample_crop_labels, num_crops=16, save_path=crops_fig_path)
    plt.close()
    print(f"Saved warped crops grid to {crops_fig_path}")

    # 3. Prepare Balanced Training Data
    print("\n[Step 3/5] Extracting Balanced Positive & Negative ROI Crops...")
    X_train, y_cls_train, y_bbox_train = prepare_rcnn_training_data(
        train_images, train_annos, neg_to_pos_ratio=3.0, seed=42
    )
    X_val, y_cls_val, y_bbox_val = prepare_rcnn_training_data(
        val_images, val_annos, neg_to_pos_ratio=3.0, seed=101
    )

    num_pos_train = int(np.sum(y_cls_train > 0))
    num_bg_train = int(np.sum(y_cls_train == 0))
    print(f"Training crops pool: {len(X_train)} (Foreground: {num_pos_train}, Background: {num_bg_train})")
    print(f"Validation crops pool: {len(X_val)}")

    # 4. Load or Train Model
    print("\n[Step 4/5] Loading or Training Stage 2 Multi-Task CNN...")
    model, history = load_or_train_rcnn_detector(
        checkpoint_path=checkpoint_path,
        X_train=X_train,
        y_cls_train=y_cls_train,
        y_bbox_train=y_bbox_train,
        X_val=X_val,
        y_cls_val=y_cls_val,
        y_bbox_val=y_bbox_val,
        force_retrain=force_retrain,
        epochs=15,
    )

    # 5. Evaluate on Test Set & Full Pipeline Visualization on Multi-Shape Scene
    print("\n[Step 5/5] Evaluating End-to-End Object Detection & NMS on Test Set...")
    eval_metrics = evaluate_dataset_map(model, test_images, test_annos, conf_thresh=0.75, nms_iou_thresh=0.2)
    print("\nTest Set Evaluation Metrics:")
    for k, v in eval_metrics.items():
        print(f"  - {k}: {v:.4f}")

    # Pick a multi-shape test scene with 3 objects (e.g. Circle, Rectangle, Triangle)
    multi_test_idx = 0
    for i, anno in enumerate(test_annos):
        if len(anno["boxes"]) >= 3 and len(np.unique(anno["classes"])) >= 2:
            multi_test_idx = i
            break

    test_img = test_images[multi_test_idx]
    test_gt_boxes = test_annos[multi_test_idx]["boxes"]
    test_gt_classes = test_annos[multi_test_idx]["classes"]

    detection_result = detect_objects_rcnn(
        model, test_img, conf_thresh=0.75, nms_iou_thresh=0.2
    )

    pipeline_fig_path = os.path.join(CURRENT_DIR, "rcnn_04_full_pipeline.png")
    visualize_complete_pipeline(
        test_img,
        test_gt_boxes,
        test_gt_classes,
        detection_result["proposals"],
        detection_result["raw_boxes"],
        detection_result["raw_scores"],
        detection_result["raw_classes"],
        detection_result["nms_boxes"],
        detection_result["nms_scores"],
        detection_result["nms_classes"],
        save_path=pipeline_fig_path,
    )
    plt.close()
    print(f"Saved complete multi-shape R-CNN pipeline figure to {pipeline_fig_path}")

    print("\n" + "=" * 70)
    print(" EXPERIMENT 09 COMPLETED SUCCESSFULLY!")
    print(f" Checkpoint stored at: {checkpoint_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiment 09: Simple R-CNN")
    parser.add_argument("--retrain", action="store_true", help="Force model retraining even if checkpoint exists")
    args = parser.parse_args()

    run_experiment(force_retrain=args.retrain)
