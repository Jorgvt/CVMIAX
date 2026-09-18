"""Training pipeline for Full-Reference and No-Reference IQA models on TID2008.

Includes SROCC & PLCC tracking callback, loss convergence monitoring,
and weights persistence.
"""

import os
from typing import Dict, Tuple
import keras
import numpy as np
import tensorflow as tf

try:
    from .dataset import (
        build_tf_datasets,
        load_tid2008_raw,
        prepare_numpy_arrays,
        split_tid2008_by_reference,
    )
    from .metrics import compute_plcc, compute_srocc
    from .models import build_fr_iqa_model, build_nr_iqa_model
except ImportError:
    from dataset import (
        build_tf_datasets,
        load_tid2008_raw,
        prepare_numpy_arrays,
        split_tid2008_by_reference,
    )
    from metrics import compute_plcc, compute_srocc
    from models import build_fr_iqa_model, build_nr_iqa_model



class IQAMetricsCallback(keras.callbacks.Callback):
    """Custom callback to compute Spearman and Pearson rank correlations at each epoch end."""

    def __init__(
        self,
        val_inputs,
        val_targets: np.ndarray,
        mode: str = "full_reference",
    ):
        super().__init__()
        self.val_inputs = val_inputs
        self.val_targets = val_targets
        self.mode = mode
        self.srocc_history = []
        self.plcc_history = []

    def on_epoch_end(self, epoch: int, logs=None):
        logs = logs or {}
        if self.mode == "full_reference":
            preds = self.model.predict(
                {"reference": self.val_inputs[0], "distorted": self.val_inputs[1]},
                verbose=0,
            ).flatten()
        else:
            preds = self.model.predict(self.val_inputs, verbose=0).flatten()

        val_srocc = compute_srocc(self.val_targets, preds)
        val_plcc = compute_plcc(self.val_targets, preds, apply_logistic_fit=False)

        self.srocc_history.append(val_srocc)
        self.plcc_history.append(val_plcc)

        logs["val_srocc"] = val_srocc
        logs["val_plcc"] = val_plcc
        print(f" - val_srocc: {val_srocc:.4f} - val_plcc: {val_plcc:.4f}")


def train_models(
    target_size: Tuple[int, int] = (224, 224),
    epochs: int = 25,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weights_dir: str = "weights",
    save_weights: bool = True,
    force_retrain: bool = False,
) -> Tuple[keras.Model, keras.Model, Dict, Dict, Tuple]:
    """Train or load both FR-IQA and NR-IQA models on TID2008 dataset.

    Returns:
        fr_model, nr_model, fr_history, nr_history, dataset_bundle
    """
    os.makedirs(weights_dir, exist_ok=True)
    fr_weights_path = os.path.join(weights_dir, "fr_iqa_model.weights.h5")
    nr_weights_path = os.path.join(weights_dir, "nr_iqa_model.weights.h5")

    print("Loading TID2008 dataset from Hugging Face...")
    raw_data = load_tid2008_raw()

    train_data, val_data, test_data = split_tid2008_by_reference(raw_data)
    print(
        f"Reference-Independent Split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}"
    )

    print("Extracting preprocessed image arrays...")
    tr_refs, tr_dists, tr_mos, tr_rids, tr_dids, tr_ints = prepare_numpy_arrays(
        train_data, target_size=target_size
    )
    val_refs, val_dists, val_mos, val_rids, val_dids, val_ints = prepare_numpy_arrays(
        val_data, target_size=target_size
    )
    te_refs, te_dists, te_mos, te_rids, te_dids, te_ints = prepare_numpy_arrays(
        test_data, target_size=target_size
    )

    fr_model = build_fr_iqa_model(input_shape=(target_size[0], target_size[1], 3))
    nr_model = build_nr_iqa_model(input_shape=(target_size[0], target_size[1], 3))

    fr_history = {"loss": [], "val_loss": [], "val_srocc": [], "val_plcc": []}
    nr_history = {"loss": [], "val_loss": [], "val_srocc": [], "val_plcc": []}

    if not force_retrain and os.path.exists(fr_weights_path) and os.path.exists(nr_weights_path):
        print(f"\nPre-trained weights found in '{weights_dir}'. Loading saved checkpoints...")
        fr_model.load_weights(fr_weights_path)
        nr_model.load_weights(nr_weights_path)
        print("Loaded weights successfully! (Set force_retrain=True to train from scratch)")
    else:
        # 1. Train Full-Reference Model
        print("\n" + "=" * 50)
        print("Training Full-Reference (FR-IQA) Siamese Difference CNN...")
        print("=" * 50)
        fr_train_ds, fr_val_ds, _ = build_tf_datasets(
            (tr_refs, tr_dists, tr_mos),
            (val_refs, val_dists, val_mos),
            (te_refs, te_dists, te_mos),
            batch_size=batch_size,
            mode="full_reference",
        )

        fr_optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        fr_model.compile(optimizer=fr_optimizer, loss="huber", metrics=["mae"])

        fr_callback = IQAMetricsCallback(
            val_inputs=(val_refs, val_dists),
            val_targets=val_mos,
            mode="full_reference",
        )

        fr_lr_schedule = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=1
        )

        fr_hist = fr_model.fit(
            fr_train_ds,
            validation_data=fr_val_ds,
            epochs=epochs,
            callbacks=[fr_callback, fr_lr_schedule],
            verbose=1,
        )
        fr_history = fr_hist.history
        fr_history["val_srocc"] = fr_callback.srocc_history
        fr_history["val_plcc"] = fr_callback.plcc_history

        if save_weights:
            fr_model.save_weights(fr_weights_path)
            print(f"Saved FR-IQA weights to {fr_weights_path}")

        # 2. Train No-Reference Model
        print("\n" + "=" * 50)
        print("Training No-Reference (NR-IQA) Blind CNN...")
        print("=" * 50)
        nr_train_ds, nr_val_ds, _ = build_tf_datasets(
            (tr_refs, tr_dists, tr_mos),
            (val_refs, val_dists, val_mos),
            (te_refs, te_dists, te_mos),
            batch_size=batch_size,
            mode="no_reference",
        )

        nr_optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        nr_model.compile(optimizer=nr_optimizer, loss="huber", metrics=["mae"])

        nr_callback = IQAMetricsCallback(
            val_inputs=val_dists,
            val_targets=val_mos,
            mode="no_reference",
        )

        nr_lr_schedule = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=1
        )

        nr_hist = nr_model.fit(
            nr_train_ds,
            validation_data=nr_val_ds,
            epochs=epochs,
            callbacks=[nr_callback, nr_lr_schedule],
            verbose=1,
        )
        nr_history = nr_hist.history
        nr_history["val_srocc"] = nr_callback.srocc_history
        nr_history["val_plcc"] = nr_callback.plcc_history

        if save_weights:
            nr_model.save_weights(nr_weights_path)
            print(f"Saved NR-IQA weights to {nr_weights_path}")

    dataset_bundle = {
        "train": (tr_refs, tr_dists, tr_mos, tr_rids, tr_dids, tr_ints),
        "val": (val_refs, val_dists, val_mos, val_rids, val_dids, val_ints),
        "test": (te_refs, te_dists, te_mos, te_rids, te_dids, te_ints),
    }

    return fr_model, nr_model, fr_history, nr_history, dataset_bundle



if __name__ == "__main__":
    train_models(epochs=10)
