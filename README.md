# CVMIAX: Computer Vision & Deep Learning Experiments

A pedagogical repository containing experiments, proofs, and visualizations showcasing fundamental phenomena in deep learning computer vision architectures using **Keras** and **uv**.

---

## Experiments Index

### [01. Gradient Propagation & Vanishing Gradients: Plain VGG vs ResNet](file:///Users/jorgvt/Developer/CVMIAX/experiments/01_gradient_flow_vgg_vs_resnet/README.md)
- **Topic**: Demonstrates why deep feedforward sequential CNNs suffer from the vanishing gradient / degradation problem, and how residual shortcut connections $\mathcal{F}(x) + x$ create an additive gradient highway enabling arbitrary depth.
- **Key Visualizations**:
  - Layer-wise gradient norm decay (Log and Linear scales).
  - First-layer gradient magnitude collapse vs. network depth.
  - Training loss and validation accuracy curves showcasing the degradation problem on CIFAR-10.
- **Code**:
  - Models: [`models.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/01_gradient_flow_vgg_vs_resnet/models.py)
  - Gradient Analysis: [`gradient_analysis.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/01_gradient_flow_vgg_vs_resnet/gradient_analysis.py)
  - Training Comparison: [`train_comparison.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/01_gradient_flow_vgg_vs_resnet/train_comparison.py)
  - Master Runner: [`run_all.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/01_gradient_flow_vgg_vs_resnet/run_all.py)

---

### [02. Data Augmentation Guide: Good Defaults & Label-Altering Pitfalls](file:///Users/jorgvt/Developer/CVMIAX/experiments/02_data_augmentation_guide/README.md)
- **Topic**: Visual reference guide designed for course slides contrasting safe default augmentations (crops, flips, color jitter, Mixup/CutMix, geometric sync) with failure modes that alter semantic ground truth labels (text/anatomy flipping, missing small objects, diagnostic color destruction, extreme non-rigid distortion).
- **Key Slide Assets**:
  - [Good Defaults Slide Poster](file:///Users/jorgvt/Developer/CVMIAX/experiments/02_data_augmentation_guide/figures/slide_good_defaults_overview.png)
  - [Pitfalls & Failure Modes Slide Poster](file:///Users/jorgvt/Developer/CVMIAX/experiments/02_data_augmentation_guide/figures/slide_pitfalls_overview.png)
- **Code**:
  - Visualization Generator: [`generate_visualizations.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/02_data_augmentation_guide/generate_visualizations.py)

---

### [03. Self-Supervised Learning Pretext Tasks: Rotation, Jigsaw, and Colorization](file:///Users/jorgvt/Developer/CVMIAX/experiments/03_self_supervised_pretext_tasks/README.md)
- **Topic**: Visual and algorithmic walkthrough of how self-supervised pretext tasks automatically construct inputs and supervisory target labels from unlabeled images, and the visual inductive biases each task instills in the representation.
- **Key Slide Assets**:
  - [Master SSL Pretext Tasks Overview Poster](file:///Users/jorgvt/Developer/CVMIAX/experiments/03_self_supervised_pretext_tasks/figures/slide_ssl_pretext_tasks_overview.png)
  - [Rotation Prediction Task](file:///Users/jorgvt/Developer/CVMIAX/experiments/03_self_supervised_pretext_tasks/figures/01_rotation_prediction_task.png)
  - [Jigsaw Puzzle Solving Task](file:///Users/jorgvt/Developer/CVMIAX/experiments/03_self_supervised_pretext_tasks/figures/02_jigsaw_puzzle_task.png)
  - [Image Colorization Task](file:///Users/jorgvt/Developer/CVMIAX/experiments/03_self_supervised_pretext_tasks/figures/03_colorization_task.png)
- **Code**:
  - Data Generators: [`data_generators.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/03_self_supervised_pretext_tasks/data_generators.py)
  - Architectures: [`models.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/03_self_supervised_pretext_tasks/models.py)
  - Visualizer: [`visualize_tasks.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/03_self_supervised_pretext_tasks/visualize_tasks.py)
  - Training Demo: [`train_pretext_demo.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/03_self_supervised_pretext_tasks/train_pretext_demo.py)
  - Master Runner: [`run_all.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/03_self_supervised_pretext_tasks/run_all.py)

---

### [04. Embedding Space Collapse in Self-Supervised Learning](file:///Users/jorgvt/Developer/CVMIAX/experiments/04_embedding_space_collapse/README.md)
- **Topic**: Visual and mathematical examination of complete constant collapse vs. dimensional subspace collapse in Siamese representation learning, diagnostic metrics (Covariance heatmaps, SVD singular value spectra), and post-training weight zeroing.
- **Key Slide Assets**:
  - [Master Collapse Overview Poster](file:///Users/jorgvt/Developer/CVMIAX/experiments/04_embedding_space_collapse/figures/slide_embedding_collapse_overview.png)
  - [2D Latent Space Comparison](file:///Users/jorgvt/Developer/CVMIAX/experiments/04_embedding_space_collapse/figures/01_2d_embedding_space_comparison.png)
  - [Spectral & Covariance Matrix Diagnostics](file:///Users/jorgvt/Developer/CVMIAX/experiments/04_embedding_space_collapse/figures/02_spectral_analysis_and_covariance.png)
  - [Post-Training Weight Inspection](file:///Users/jorgvt/Developer/CVMIAX/experiments/04_embedding_space_collapse/figures/05_weight_distributions_and_norms.png)
  - [Empirical Siamese Training Curves](file:///Users/jorgvt/Developer/CVMIAX/experiments/04_embedding_space_collapse/figures/04_empirical_collapse_curves.png)
- **Code**:
  - Collapse Utilities: [`collapse_utils.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/04_embedding_space_collapse/collapse_utils.py)
  - Visualizer: [`visualize_collapse.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/04_embedding_space_collapse/visualize_collapse.py)
  - Empirical Siamese Training: [`train_siamese_collapse_demo.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/04_embedding_space_collapse/train_siamese_collapse_demo.py)
  - Master Runner: [`run_all.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/04_embedding_space_collapse/run_all.py)

---

### [05. Masked Autoencoders (MAE) for Vision (He et al., 2022)](file:///Users/jorgvt/Developer/CVMIAX/experiments/05_masked_autoencoders_mae/README.md)
- **Topic**: Demonstrates the masking mechanism (patch partitioning and 75% random masking), asymmetric ViT encoder-decoder computation, exact information routing, and qualitative inpainting reconstructions.
- **Key Slide Assets**:
  - [Master MAE Overview Poster](file:///Users/jorgvt/Developer/CVMIAX/experiments/05_masked_autoencoders_mae/figures/slide_mae_overview.png)
  - [Masking Mechanism & Information Flow](file:///Users/jorgvt/Developer/CVMIAX/experiments/05_masked_autoencoders_mae/figures/01_mae_patch_masking_mechanism.png)
  - [Masking Ratio Comparison (25% vs 50% vs 75% vs 90%)](file:///Users/jorgvt/Developer/CVMIAX/experiments/05_masked_autoencoders_mae/figures/02_mae_reconstruction_vs_mask_ratio.png)
  - [Information Breakdown (Encoder vs Decoder vs Loss)](file:///Users/jorgvt/Developer/CVMIAX/experiments/05_masked_autoencoders_mae/figures/03_information_used_by_model.png)
  - [CIFAR-10 Inpainting Reconstructions](file:///Users/jorgvt/Developer/CVMIAX/experiments/05_masked_autoencoders_mae/figures/04_mae_cifar_reconstructions.png)
- **Code**:
  - Patch & Mask Utilities: [`mae_utils.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/05_masked_autoencoders_mae/mae_utils.py)
  - ViT-MAE Architecture: [`models.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/05_masked_autoencoders_mae/models.py)
  - Visualizer: [`visualize_mae.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/05_masked_autoencoders_mae/visualize_mae.py)
  - Training Demo: [`train_mae_demo.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/05_masked_autoencoders_mae/train_mae_demo.py)
  - Master Runner: [`run_all.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/05_masked_autoencoders_mae/run_all.py)

---

### [06. Deep Computer Vision Model Architectures: ResNet, Plain CNN, Inception, EfficientNet, ConvNeXt](file:///Users/jorgvt/Developer/CVMIAX/experiments/06_resnet_architecture/README.md)
- **Topic**: Modular implementations of 5 computer vision architectures in Keras, isolating residual learning ($y = \mathcal{F}(x) + x$), multi-scale Inception branches, MBConv + Squeeze-and-Excitation attention, and modern ConvNeXt design principles, complete with independent training scripts and stored pre-trained weights (`.weights.h5`).
- **Key Slide Assets**:
  - [Master Architecture Poster (5 Models)](file:///Users/jorgvt/Developer/CVMIAX/experiments/06_resnet_architecture/figures/00_all_architectures_overview.png)
  - [Architecture Contrast: Plain Block vs Residual Block](file:///Users/jorgvt/Developer/CVMIAX/experiments/06_resnet_architecture/figures/01_architecture_plain_vs_resnet.png)
  - [Training Dynamics & Accuracy Curves](file:///Users/jorgvt/Developer/CVMIAX/experiments/06_resnet_architecture/figures/02_training_dynamics_comparison.png)
  - [Layerwise Weight L2 Norms & Distribution](file:///Users/jorgvt/Developer/CVMIAX/experiments/06_resnet_architecture/figures/03_weight_distributions.png)
- **Code**:
  - ResNet Architecture & Weights: [`resnet.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/06_resnet_architecture/resnet.py)
  - Plain CNN Architecture & Weights: [`plain_cnn.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/06_resnet_architecture/plain_cnn.py)
  - EfficientNet Architecture & Weights: [`efficientnet.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/06_resnet_architecture/efficientnet.py)
  - ConvNeXt Architecture & Weights: [`convnext.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/06_resnet_architecture/convnext.py)
  - Inception Architecture & Weights: [`inception.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/06_resnet_architecture/inception.py)
  - Independent Training Scripts: [`train_resnet.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/06_resnet_architecture/train_resnet.py), [`train_plain_cnn.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/06_resnet_architecture/train_plain_cnn.py), [`train_efficientnet.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/06_resnet_architecture/train_efficientnet.py), [`train_convnext.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/06_resnet_architecture/train_convnext.py), [`train_inception.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/06_resnet_architecture/train_inception.py)
  - Comparison & Evaluation: [`compare.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/06_resnet_architecture/compare.py)
  - Dataset & Augmentation Pipeline: [`dataset.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/06_resnet_architecture/dataset.py)
  - Visualizer: [`visualize.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/06_resnet_architecture/visualize.py)
  - Master CLI Runner: [`run_all.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/06_resnet_architecture/run_all.py)

---

### [07. Candlestick Pattern Detection with Modern Object Detectors](file:///Users/jorgvt/Developer/CVMIAX/experiments/07_candles/README.md)
- **Topic**: Single-stage Object Detection (YOLO) applied to financial candlestick charts with deterministic rule-based auto-labeling (TA-Lib equivalent rules), bounding box coordinate math, and Sim-to-Real domain shift analysis across trading platforms.
- **Key Assets**:
  - Sample prediction overlays with ground truth vs. YOLO bounding boxes.
  - Domain shift robustness benchmark (clean, dark theme, gridlines, volume, noise).
  - Real broker screenshot generalization tests in `assets/real_charts/`.
- **Code**:
  - Pattern Recognizer Rules: [`patterns.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/07_candles/patterns.py)
  - Chart Renderer & BBox Engine: [`renderer.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/07_candles/renderer.py)
  - Dataset Generator: [`data_generator.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/07_candles/data_generator.py)
  - YOLO Training: [`train.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/07_candles/train.py)
  - Evaluation & Domain Shift: [`evaluate.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/07_candles/evaluate.py)
  - Master CLI Runner: [`run_all.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/07_candles/run_all.py)

---

### [10. Plain FCN vs U-Net: The Power of Skip Connections in Semantic Segmentation](file:///Users/jorgvt/Developer/CVMIAX/experiments/10_fcn_vs_unet_segmentation/README.md)
- **Topic**: Pedagogical comparison isolating the role of lateral skip connections in semantic segmentation, showing why bottleneck-only architectures (Plain FCN) produce blurry boundaries and rounded corners while U-Net achieves razor-sharp edge precision. Includes automated checkpointing (`.weights.h5`) to prevent redundant training.
- **Key Assets**:
  - Architectural schematic: Plain FCN (Bottleneck) vs U-Net (Skip Highways).
  - Multi-sample qualitative comparison with misclassification error maps.
  - Quantitative Boundary IoU vs. Mean IoU analysis and 1D edge transition sharpness profiles.
- **Code**:
  - Dataset Generator: [`dataset.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/10_fcn_vs_unet_segmentation/dataset.py)
  - Architectures: [`models.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/10_fcn_vs_unet_segmentation/models.py)
  - Training Pipeline: [`train.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/10_fcn_vs_unet_segmentation/train.py)
  - Evaluation Metrics: [`evaluate.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/10_fcn_vs_unet_segmentation/evaluate.py)
  - Visualizer: [`visualize.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/10_fcn_vs_unet_segmentation/visualize.py)
  - Master Runner: [`run_all.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/10_fcn_vs_unet_segmentation/run_all.py)

---

### [11. Neural Radiance Fields (NeRF): 3D Novel View Synthesis & Volumetric Rendering](file:///Users/jorgvt/Developer/CVMIAX/experiments/11_nerf_novel_view_synthesis/README.md)
- **Topic**: Minimal, pedagogical end-to-end implementation of Neural Radiance Fields in Keras/TensorFlow. Covers pinhole camera ray casting, stratified 3D sampling, overcoming spectral bias with Fourier positional embeddings, differentiable volumetric rendering with accumulated transmittance, and 360-degree orbital novel view synthesis.
- **Key Assets**:
  - 3D camera ray casting and stratified sampling geometry visualizer.
  - Spectral bias ablation: Raw coordinates ($L=0$, oversmoothed) vs. Fourier features ($L=6$, sharp).
  - Multi-modal evaluation: Ground Truth RGB vs. Rendered RGB vs. Depth Map vs. Accumulated Opacity.
  - 360-degree orbital camera novel view synthesis animation (`nerf_novel_views_360.gif`).
- **Code**:
  - Dataset & Procedural Fallback: [`dataset.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/11_nerf_novel_view_synthesis/dataset.py)
  - Ray Casting & Stratified Sampling: [`rays.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/11_nerf_novel_view_synthesis/rays.py)
  - Positional Encoding & Radiance MLP: [`models.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/11_nerf_novel_view_synthesis/models.py)
  - Volumetric Rendering Engine: [`rendering.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/11_nerf_novel_view_synthesis/rendering.py)
  - Training & PSNR Tracking: [`train.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/11_nerf_novel_view_synthesis/train.py)
  - Evaluation & Novel Trajectories: [`evaluate.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/11_nerf_novel_view_synthesis/evaluate.py)
  - Visualizer: [`visualize.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/11_nerf_novel_view_synthesis/visualize.py)
  - Master Runner: [`run_all.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/11_nerf_novel_view_synthesis/run_all.py)

---

### [12. 3D Gaussian Splatting (3DGS): Real-Time Novel View Synthesis](file:///Users/jorgvt/Developer/CVMIAX/experiments/12_3d_gaussian_splatting/README.md)
- **Topic**: Minimal, self-contained, pure-Python/TensorFlow implementation of the full 3D Gaussian Splatting pipeline from scratch without external CUDA/C++ compilers. Covers explicit 3D Gaussian parameterization (scaling vectors and quaternions), projective camera Jacobian $J$, EWA 2D covariance projection, depth-sorted front-to-back alpha compositing ("Over" operator), multi-view photometric optimization, and 360-degree orbital novel view rendering.
- **Key Assets**:
  - 3D Gaussian spatial cloud and camera pose visualizer.
  - Convergence diagnostics: Training loss, L1 error, and PSNR over iterations.
  - Qualitative evaluation: Ground Truth vs. 3DGS Render vs. Absolute Residual Error.
  - 360-degree orbital novel view synthesis animation (`3dgs_novel_views_360.gif`).
- **Code**:
  - Dataset & Orbital Trajectories: [`dataset.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/12_3d_gaussian_splatting/dataset.py)
  - 3D Gaussian Representation: [`models.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/12_3d_gaussian_splatting/models.py)
  - Differentiable EWA Splatting: [`splatting.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/12_3d_gaussian_splatting/splatting.py)
  - Photometric Optimization: [`train.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/12_3d_gaussian_splatting/train.py)
  - Evaluation & Novel Views: [`evaluate.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/12_3d_gaussian_splatting/evaluate.py)
  - Visualizer & GIF Generator: [`visualize.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/12_3d_gaussian_splatting/visualize.py)
  - Master Runner: [`run_all.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/12_3d_gaussian_splatting/run_all.py)

---

### [13. Traditional Computer Vision Pipeline: Preprocessing → Features → Feature Processing → Classifier](file:///Users/jorgvt/Developer/CVMIAX/experiments/13_traditional_cv_pipeline/README.md)
- **Topic**: Pedagogical decomposition of the classic 4-stage computer vision pipeline before deep learning. Solves image classification from scratch using moments-based deskewing, Gaussian denoising, hand-crafted feature extractors (HOG gradient histograms, LBP micro-texture codes, Hu moment invariants), PCA eigen-decomposition & scree variance analysis, and statistical classifiers (Linear & RBF SVM, Random Forest, k-NN).
- **Key Visualizations**:
  - 4-Stage Pipeline Schematic ([`01_pipeline_stages_overview.png`](file:///Users/jorgvt/Developer/CVMIAX/experiments/13_traditional_cv_pipeline/01_pipeline_stages_overview.png)).
  - Hand-Crafted Feature Representations Across Classes ([`02_feature_extraction_deep_dive.png`](file:///Users/jorgvt/Developer/CVMIAX/experiments/13_traditional_cv_pipeline/02_feature_extraction_deep_dive.png)).
  - Feature Space Disentanglement: Raw Pixels vs HOG 2D PCA & Scree Variance Plot ([`03_pca_feature_space_and_variance.png`](file:///Users/jorgvt/Developer/CVMIAX/experiments/13_traditional_cv_pipeline/03_pca_feature_space_and_variance.png)).
  - Multi-Pipeline Benchmark Comparison & Confusion Matrices ([`04_classifier_benchmark_comparison.png`](file:///Users/jorgvt/Developer/CVMIAX/experiments/13_traditional_cv_pipeline/04_classifier_benchmark_comparison.png)).
  - Qualitative Success Gallery & Hand-Crafted Feature Failure Modes ([`05_sample_predictions_and_failure_analysis.png`](file:///Users/jorgvt/Developer/CVMIAX/experiments/13_traditional_cv_pipeline/05_sample_predictions_and_failure_analysis.png)).
- **Code**:
  - Preprocessing (Moments Deskewing & Filters): [`preprocessing.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/13_traditional_cv_pipeline/preprocessing.py)
  - Hand-Crafted Features (HOG, LBP, Hu Moments): [`features.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/13_traditional_cv_pipeline/features.py)
  - Feature Processing (Standardization & PCA): [`feature_processing.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/13_traditional_cv_pipeline/feature_processing.py)
  - Classifiers (Linear SVM, RBF SVM, Random Forest, k-NN): [`classifiers.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/13_traditional_cv_pipeline/classifiers.py)
  - Visualizer: [`visualize.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/13_traditional_cv_pipeline/visualize.py)
  - Benchmark Suite: [`train_and_evaluate.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/13_traditional_cv_pipeline/train_and_evaluate.py)
  - Master Runner: [`run_all.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/13_traditional_cv_pipeline/run_all.py)

---

### [14. Contrastive Language-Image Pre-Training (CLIP): Dual Encoders, InfoNCE, and Zero-Shot Transfer](file:///Users/jorgvt/Developer/CVMIAX/experiments/14_clip_multimodal_contrastive/README.md)
- **Topic**: Multimodal contrastive representation learning aligning image and natural language representations into a shared unit-hypersphere metric space. Showcases symmetric InfoNCE loss, dynamic temperature scaling $\tau$, cross-modal bidirectional retrieval, compositional zero-shot binding (unseen color-shape combinations), and novel geometric shape topology evaluation (5-pointed stars and diamonds).
- **Key Visualizations**:
  - Full Architecture & InfoNCE Flowchart ([`clip_architecture_and_infonce.png`](file:///Users/jorgvt/Developer/CVMIAX/experiments/14_clip_multimodal_contrastive/figures/clip_architecture_and_infonce.png)).
  - Training Dynamics & Temperature Evolution ([`clip_training_dynamics.png`](file:///Users/jorgvt/Developer/CVMIAX/experiments/14_clip_multimodal_contrastive/figures/clip_training_dynamics.png)).
  - Cosine Similarity Alignment Heatmaps ([`clip_similarity_matrix_heatmap.png`](file:///Users/jorgvt/Developer/CVMIAX/experiments/14_clip_multimodal_contrastive/figures/clip_similarity_matrix_heatmap.png)).
  - Zero-Shot Prediction Gallery & Softmax Distribution ([`clip_zero_shot_classification.png`](file:///Users/jorgvt/Developer/CVMIAX/experiments/14_clip_multimodal_contrastive/figures/clip_zero_shot_classification.png)).
  - Text-to-Image Cross-Modal Retrieval Showcase ([`clip_cross_modal_retrieval.png`](file:///Users/jorgvt/Developer/CVMIAX/experiments/14_clip_multimodal_contrastive/figures/clip_cross_modal_retrieval.png)).
  - Shared Multimodal PCA 2D Metric Space ([`clip_joint_embedding_space.png`](file:///Users/jorgvt/Developer/CVMIAX/experiments/14_clip_multimodal_contrastive/figures/clip_joint_embedding_space.png)).
- **Code**:
  - Dataset Generator: [`dataset.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/14_clip_multimodal_contrastive/dataset.py)
  - Dual Encoders & Architecture: [`models.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/14_clip_multimodal_contrastive/models.py)
  - Loss & Metrics: [`losses.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/14_clip_multimodal_contrastive/losses.py)
  - Visualizer: [`visualize.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/14_clip_multimodal_contrastive/visualize.py)
  - Training Pipeline: [`train.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/14_clip_multimodal_contrastive/train.py)
  - Master Runner: [`run_all.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/14_clip_multimodal_contrastive/run_all.py)

---

### [15. Image Quality Assessment (IQA) on TID2008: Full-Reference vs No-Reference Deep Metrics](file:///Users/jorgvt/Developer/CVMIAX/experiments/15_image_quality_assessment_tid2008/README.md)
- **Topic**: Pedagogical implementation of Image Quality Assessment (IQA) evaluating subjective Mean Opinion Score (MOS) predictions on the TID2008 benchmark (`Jorgvt/TID2008`). Covers content-independent (reference-wise) data splitting to prevent semantic leakage, Full-Reference Siamese hierarchical feature difference networks ($\Delta \phi_k = |\phi_k(I_{\text{ref}}) - \phi_k(I_{\text{dist}})|$), No-Reference (Blind) CNN quality regression, spatial perceptual error activation maps, and standard benchmark metrics (SROCC, PLCC, KROCC, RMSE, MAE with 4-parameter logistic calibration).
- **Key Visualizations**:
  - [Distortion Taxonomy Gallery](file:///Users/jorgvt/Developer/CVMIAX/experiments/15_image_quality_assessment_tid2008/figures/01_distortion_taxonomy_gallery.png).
  - [Classical vs Perceptual Correlation Benchmark Scatter Plots](file:///Users/jorgvt/Developer/CVMIAX/experiments/15_image_quality_assessment_tid2008/figures/02_classical_vs_perceptual_correlation.png).
  - [Deep IQA Training Dynamics & Validation SROCC Progression](file:///Users/jorgvt/Developer/CVMIAX/experiments/15_image_quality_assessment_tid2008/figures/03_training_dynamics.png).
  - [SROCC Across Distortion Categories Breakdown](file:///Users/jorgvt/Developer/CVMIAX/experiments/15_image_quality_assessment_tid2008/figures/04_distortion_wise_performance_breakdown.png).
  - [Hierarchical Perceptual Error Activation Maps](file:///Users/jorgvt/Developer/CVMIAX/experiments/15_image_quality_assessment_tid2008/figures/05_perceptual_error_heatmaps.png).
- **Code**:
  - Dataset & Reference Split: [`dataset.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/15_image_quality_assessment_tid2008/dataset.py)
  - Classical & Correlation Metrics: [`metrics.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/15_image_quality_assessment_tid2008/metrics.py)
  - Siamese FR & NR Architectures: [`models.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/15_image_quality_assessment_tid2008/models.py)
  - Training Pipeline: [`train.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/15_image_quality_assessment_tid2008/train.py)
  - Benchmark Evaluator: [`evaluate.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/15_image_quality_assessment_tid2008/evaluate.py)
  - Visualizer: [`visualize.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/15_image_quality_assessment_tid2008/visualize.py)
  - Master Runner: [`run_all.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/15_image_quality_assessment_tid2008/run_all.py)

---

## Interactive Notebooks & Google Colab Access

Every experiment includes a paired master script (in Jupytext `py:percent` format) and a standalone Jupyter Notebook with 1-click **Open in Colab** badges:

| Experiment | Interactive Notebook | Google Colab |
| :--- | :--- | :--- |
| **01. Gradient Propagation (ResNet vs VGG)** | [`experiment_01_gradient_flow.ipynb`](file:///Users/jorgvt/Developer/CVMIAX/experiments/01_gradient_flow_vgg_vs_resnet/experiment_01_gradient_flow.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorgvt/CVMIAX/blob/main/experiments/01_gradient_flow_vgg_vs_resnet/experiment_01_gradient_flow.ipynb) |
| **02. Data Augmentation Guide & Pitfalls** | [`experiment_02_data_augmentation.ipynb`](file:///Users/jorgvt/Developer/CVMIAX/experiments/02_data_augmentation_guide/experiment_02_data_augmentation.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorgvt/CVMIAX/blob/main/experiments/02_data_augmentation_guide/experiment_02_data_augmentation.ipynb) |
| **03. SSL Pretext Tasks (Rotation, Jigsaw, Color)** | [`experiment_03_pretext_tasks.ipynb`](file:///Users/jorgvt/Developer/CVMIAX/experiments/03_self_supervised_pretext_tasks/experiment_03_pretext_tasks.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorgvt/CVMIAX/blob/main/experiments/03_self_supervised_pretext_tasks/experiment_03_pretext_tasks.ipynb) |
| **04. Embedding Space Collapse in SSL** | [`experiment_04_embedding_collapse.ipynb`](file:///Users/jorgvt/Developer/CVMIAX/experiments/04_embedding_space_collapse/experiment_04_embedding_collapse.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorgvt/CVMIAX/blob/main/experiments/04_embedding_space_collapse/experiment_04_embedding_collapse.ipynb) |
| **05. Masked Autoencoders (MAE)** | [`experiment_05_masked_autoencoders.ipynb`](file:///Users/jorgvt/Developer/CVMIAX/experiments/05_masked_autoencoders_mae/experiment_05_masked_autoencoders.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorgvt/CVMIAX/blob/main/experiments/05_masked_autoencoders_mae/experiment_05_masked_autoencoders.ipynb) |
| **05. ViT-MAE Training Loop Demo** | [`train_mae_demo.ipynb`](file:///Users/jorgvt/Developer/CVMIAX/experiments/05_masked_autoencoders_mae/train_mae_demo.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorgvt/CVMIAX/blob/main/experiments/05_masked_autoencoders_mae/train_mae_demo.ipynb) |
| **06. ResNet Architecture vs Plain CNN** | [`experiment_06_resnet.ipynb`](file:///Users/jorgvt/Developer/CVMIAX/experiments/06_resnet_architecture/experiment_06_resnet.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorgvt/CVMIAX/blob/main/experiments/06_resnet_architecture/experiment_06_resnet.ipynb) |
| **07. Candlestick Pattern Detection** | [`experiment_07_candlestick_detection.ipynb`](file:///Users/jorgvt/Developer/CVMIAX/experiments/07_candles/experiment_07_candlestick_detection.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorgvt/CVMIAX/blob/main/experiments/07_candles/experiment_07_candlestick_detection.ipynb) |
| **10. Plain FCN vs U-Net Segmentation** | [`experiment_10_fcn_vs_unet.ipynb`](file:///Users/jorgvt/Developer/CVMIAX/experiments/10_fcn_vs_unet_segmentation/experiment_10_fcn_vs_unet.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorgvt/CVMIAX/blob/main/experiments/10_fcn_vs_unet_segmentation/experiment_10_fcn_vs_unet.ipynb) |
| **11. NeRF (Novel View Synthesis)** | [`experiment_11_nerf.ipynb`](file:///Users/jorgvt/Developer/CVMIAX/experiments/11_nerf_novel_view_synthesis/experiment_11_nerf.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorgvt/CVMIAX/blob/main/experiments/11_nerf_novel_view_synthesis/experiment_11_nerf.ipynb) |
| **12. 3D Gaussian Splatting (3DGS)** | [`experiment_12_3d_gaussian_splatting.ipynb`](file:///Users/jorgvt/Developer/CVMIAX/experiments/12_3d_gaussian_splatting/experiment_12_3d_gaussian_splatting.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorgvt/CVMIAX/blob/main/experiments/12_3d_gaussian_splatting/experiment_12_3d_gaussian_splatting.ipynb) |
| **13. Traditional Computer Vision Pipeline** | [`experiment_13_traditional_cv_pipeline.ipynb`](file:///Users/jorgvt/Developer/CVMIAX/experiments/13_traditional_cv_pipeline/experiment_13_traditional_cv_pipeline.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorgvt/CVMIAX/blob/main/experiments/13_traditional_cv_pipeline/experiment_13_traditional_cv_pipeline.ipynb) |
| **14. Multi-Modal CLIP Contrastive Learning** | [`experiment_14_clip_multimodal_contrastive.ipynb`](file:///Users/jorgvt/Developer/CVMIAX/experiments/14_clip_multimodal_contrastive/experiment_14_clip_multimodal_contrastive.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorgvt/CVMIAX/blob/main/experiments/14_clip_multimodal_contrastive/experiment_14_clip_multimodal_contrastive.ipynb) |
| **15. Image Quality Assessment (TID2008)** | [`experiment_15_image_quality_assessment_tid2008.ipynb`](file:///Users/jorgvt/Developer/CVMIAX/experiments/15_image_quality_assessment_tid2008/experiment_15_image_quality_assessment_tid2008.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorgvt/CVMIAX/blob/main/experiments/15_image_quality_assessment_tid2008/experiment_15_image_quality_assessment_tid2008.ipynb) |

---

## Getting Started & Development Workflow

### 1. Environment Setup
Clone the repository and install all dependencies (including developer tools) using `uv`:
```bash
# Clone the repository
git clone https://github.com/jorgvt/CVMIAX.git
cd CVMIAX

# Install Python environment and dependencies
uv sync --dev
```

### 2. Install Git Pre-Commit Hooks (Automatic Notebook Sync)
To ensure that `.py` master scripts and paired `.ipynb` notebooks stay in lockstep automatically during `git commit`, install the pre-commit hook:
```bash
uv run pre-commit install
```
* With this installed, whenever you modify an `experiment_*.py` script and run `git commit`, the paired `.ipynb` notebook is automatically updated and staged for you.

### 3. Running Modular Experiment Scripts
You can run any experiment's full demonstration suite via standard CLI commands:
```bash
# Experiment 01: Gradient Propagation & Vanishing Gradients (ResNet vs VGG)
uv run python experiments/01_gradient_flow_vgg_vs_resnet/run_all.py

# Experiment 02: Data Augmentation Guide & Pitfalls
uv run python experiments/02_data_augmentation_guide/generate_visualizations.py

# Experiment 03: Self-Supervised Learning Pretext Tasks
uv run python experiments/03_self_supervised_pretext_tasks/run_all.py

# Experiment 04: Embedding Space Collapse in SSL
uv run python experiments/04_embedding_space_collapse/run_all.py

# Experiment 05: Masked Autoencoders (MAE)
uv run python experiments/05_masked_autoencoders_mae/run_all.py

# Experiment 07: Candlestick Pattern Detection
uv run python experiments/07_candles/run_all.py

# Experiment 15: Image Quality Assessment on TID2008
uv run python experiments/15_image_quality_assessment_tid2008/run_all.py
```

```

### 4. Interactive Notebooks & Jupytext Synchronization
* **Interactive Editing**: You can open and edit the master `.py` scripts directly in VS Code (with native cell code lenses `# %%`) or JupyterLab.
* **Manual Synchronization**: To manually sync all paired scripts and notebooks across the repository:
  ```bash
  uv run jupytext --sync experiments/**/*.py
  ```
* **Run Pre-Commit Checks Across All Files**:
  ```bash
  uv run pre-commit run --all-files
  ```
* **CI Verification**: GitHub Actions automatically runs `jupytext --sync` and verifies zero uncommitted diffs on every Pull Request.

