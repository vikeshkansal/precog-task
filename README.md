# Computer Vision Project: The Lazy Artist (Bias & Robustness)

This project explores the issue of **spurious correlations** in computer vision models using a custom "Colored MNIST" dataset. It demonstrates how standard models fail to generalize when training data has strong color-digit correlations (the "lazy" path) and explores methods to mitigate this bias.

## Project Overview

The notebook `main.ipynb` walks through the entire pipeline:
1.  **Dataset Creation**: Generating a biased Colored MNIST dataset where digit color correlates with the label (e.g., 0 is Red) in training, but is randomized in testing.
2.  **Baseline Training**: Training a simple CNN (`BaselineNet`) that achieves high training accuracy but fails completely (<5%) on the hard test set.
3.  **Interpretability**: Using techniques like Feature Visualization, Automated Captioning (BLIP), and Counterfactual Ablation to prove the model is looking at color, not shape.
4.  **Mitigation**: Implementing strategies like **Just Train Twice (JTT)** and **Unsupervised Clustering** to force the model to learn robust shape features (achieving ~85% accuracy).
5.  **Analysis**: Comparing robust vs. non-robust models using Adversarial Attacks (PGD) and Singular Learning Theory (LLC estimation).

## Requirements

The project uses Python 3. To run the notebook, you need the following libraries installed.

It is recommended to use a virtual environment.

### Core Dependencies
*   `numpy`
*   `matplotlib`
*   `torch` (PyTorch)
*   `torchvision`
*   `scikit-learn`
*   `seaborn`
*   `networkx`
*   `Pillow` (PIL)
*   `opencv-python` (cv2)
*   `tqdm`

### Advanced Analysis Dependencies
*   `transformers` (for BLIP captioning)
*   `open_clip_torch` (for CLIP analysis - imported as `open_clip`)
*   `devinterp` (for Singular Learning Theory / LLC estimation)

### Installation

You can install the dependencies using pip:

```bash
pip install numpy matplotlib torch torchvision scikit-learn seaborn networkx Pillow opencv-python tqdm transformers open_clip_torch devinterp
```

*Note: For `torch`, ensure you install the version appropriate for your hardware (CPU or CUDA).*

## Setup

1.  **Project Structure**: Ensure your directory looks like this:
    ```
    precog_cv/
    ├── main.ipynb          # The main notebook containing all code
    ├── data/               # Directory for storing MNIST data (will be created automatically)
    ├── colored_mnist/      # Directory for generated colored datasets (will be created)
    └── trained_model.pth   # Saved model weights (generated during training if not present)
    ```

2.  **Dataset**: The notebook automatically downloads the standard MNIST dataset into the `./data` folder and generates the synthetic "Colored MNIST" variants in `./colored_mnist`. No manual download is required.

## How to Run

1.  **Open `main.ipynb`**: Navigate to the file in the Jupyter interface.
2.  **Run All Cells**: You can run the cells sequentially.
    *   **Task 0**: Data generation (this takes a few minutes initially to create the colored variants).
    *   **Task 1**: Baseline training.
    *   **Task 2 & 3**: Visualization and Interpretability (these steps load the trained baseline model).
    *   **Task 4**: Mitigation training (JTT and Clustering). This involves training multiple models and will take longer (10-20 mins depending on hardware).
    *   **Task 5**: Robustness checks and SLT analysis.

## Hardware Recommendation

Training can be computationally intensive. A GPU is highly recommended. The notebook automatically detects if CUDA is available (`device = "cuda"`).

## Results Summary

*   **Baseline**: ~3.7% accuracy on randomized test set (Color biased).
*   **Robust (JTT)**: ~87% accuracy on randomized test set (Shape learned).
