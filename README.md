# PointNet Lite - ModelNet10 Demo

This repository provides a Jupyter Notebook that trains a simple PointNet Lite model on the ModelNet10 dataset using TensorFlow 2. The notebook covers:

* **Automatic download** of the ModelNet10 dataset from the official Princeton site.
* **Mesh loading & point sampling**: Samples 1,024 points from each 3D mesh (`.off` files) using `trimesh`.
* **Data normalization**: Centers each point cloud and scales to a unit sphere.
* **TensorFlow data pipeline**: Efficient batching and prefetching for GPU training.
* **PointNet Lite architecture**: A lightweight MLP-style network with pointwise convolutions and global max pooling.
* **GPU support**: Automatic detection and setup for GPU execution.
* **Training & evaluation**: Model training, validation, and test set evaluation with accuracy and loss metrics.

## Repository Structure

```
├── PointNet_Lite_ModelNet10.ipynb   # Main Jupyter Notebook
├── README.md                        # This file
└── requirements.txt                 # Python dependencies
```

## Installation

Make sure you have Python 3.8+ installed. Then install dependencies:

```bash
pip install -r requirements.txt
```

The `requirements.txt` should include:

```
tensorflow
numpy
matplotlib
trimesh
scikit-learn
tqdm
requests
```

## Usage

1. Open `PointNet_Lite_ModelNet10.ipynb` in Jupyter or Colab.
2. Run each cell in order:

   * Dataset download & extraction
   * Point cloud preparation
   * Model definition & compilation
   * Training on GPU (if available)
   * Evaluation and sample predictions
3. Adjust hyperparameters (e.g., `EPOCHS`, `BATCH_SIZE`, `NUM_POINTS`) in the notebook as needed.

## GPU Configuration

The notebook detects available GPUs via TensorFlow and enables memory growth to prevent allocation errors:

```python
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
# ... setup memory growth ...
```

Training will automatically run on `/GPU:0` if detected, otherwise on CPU.

## Results

* **Training curves**: Accuracy vs. Epochs plotted in the notebook.
* **Test accuracy**: Displayed at the end of the evaluation cell.
* **Sample predictions**: Ground-truth vs. predicted class labels printed.
