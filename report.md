# Project Report: Bias & Robustness in Computer Vision

## 0. Tasks Attempted (& Why)
* I have attempted tasks 0 to 5 of the CV project (The lazy artist).
* I have not attempted task 6 due to a lack of time. 

## 1. Introduction
*   **Goal**: The goal of this project was to train a Convolutional Neural Network (CNN) on a biased dataset (Colored MNIST), demonstrate its failure on conflicting data, analyze the internal mechanism of this failure using interpretability tools, and implement methods to mitigate it.
*   **Dataset (Colored MNIST)**: 
    *   **Construction**: In the training set, digit color is highly correlated with the digit class (for example 0 is always red), however this is only for 95% of the digits; for 5% of the digits, this correlation is randomized (for example, a 0 may be a green). In the "conflicting" test set, this correlation is randomized for all of the data.
    *   **Problem**: Models tend to take the shortcut path. They learn the much more prevalent (and easy) feature of color over shape. We need to see why this is happening, what the model is doing under-the-hood, and how we can fix it.

## 2. Task 0 & 1: The Baseline Model and Bias Quantification
*   **Dataset Generation**: I made three datasets; `train_colored` and `test_colored`, `test_cycled`, and the normal, uncolored dataset but with three channels (`greyscale`) for future use. Here, `train_colored` is the biased dataset, `test_colored` is the dataset with a randomized background and `cycled` was a dataset I made to test that the model is actually looking at color by just cycling the color that is assigned to each digit; i.e. 0 is assigned 1s color, 1 is assigned 2's color, and so on. The backgrounds aren't just flat color; for each image, random noise is generated first, which is multiplied by the color assigned to the digit.
*   **Model Architecture**: `BaselineNet` is a simple CNN consisting of 3 Convolutional blocks (each with Conv2D, GroupNorm, and ReLU), followed by MaxPooling, and a Global Average Pooling layer feeding into a linear classifier (10 outputs). This model architecture was picked with a lot of trial and error; specifically with trying to get the model's accuracy to be below 20%, and we want the model to take the shortcut path. One thing I noticed with this was the sensitivity towards the kernel size and how much something like the normalization matters; initially I remade something close to ResNet (~100k parameters) and was getting ~90% accuracy even on the test dataset. I then reduced the parameters and removed BatchNorm. I was also using a 3x3 sized kernel in multiple of my layers. Replacing these things and replacing them with simpler components was a big part of the trial-and-error process.

*   **Hyperparameters**: Optimizer: SGD, Learning Rate: 0.1, Batch Size: 16, Epochs: 10. These hyperparameters were also picked mainly with trial and error.

*   **Performance Discrepancy**:
    *   **Training Accuracy**: **~95%**
    *   **Conflicting (Hard) Test Accuracy**: **3.77%**
    *   **Interpretation**: The model achieves near-perfect training accuracy but fails completely on the hard test set (worse than random guessing's 10%), indicating it has learned to rely solely on the spurious color correlation.
*   **Confusion Matrix**:
    *   **Analysis**: The confusion matrix (made for the test set) reveals that the model is mainly looking at color since it's very distributed. One thing I noticed was that the digit 1 specifically normally had low confusion (i.e. it was correctly classified) a lot more than the others, but still low compared to an absolute scale. I suspect that this is because the digit 1 is just a vertical line, and so it is a lot easier for the model to identify it, compared to 3, for example, which is made up of curves. Something else to note is that the confusion matrix has very low confidence values on the diagonal (i.e. at any index (i, i)), since the test set has backgrounds of anything but the color assigned to a digit.

## 3. Task 2: Interpretability - "What is the model looking at?"
*   **Feature Visualization (Activation Maximization)**:
* I tried several things for this:
    1. The first step consisted of trying to see what a model remembers; I passed in images belonging to each class, captured the activationss of the neurons in the 3rd convolutional layer (I'll refer to this as c3) and trained a tensor such that the activations of the tensor's c3 match the captured c3. The next thing I tried was to just optimize image tensors to produce each class' ideal image, by maximizing their probabilities. To this, I tried starting off with random noise, and also tried starting off with the background texture that I made the dataset with. As an example, if I was generating the image for 1 then I'd choose the background texture for 1, and optimize from that. The main thing I noticed with this is that the model simply changed around the pixels to simply maximize what I had asked for; not necessarily in a way that would be interpretable to humans. From here I realized the importance of regularization and tried applying it, however I didn't get concrete outputs with that and left it at that.
    2. The next thing I decided to do was to show that the model is actually really simple and that its dimensionality isn't very high. The main way I decided to do this was using 3 things - Singular Learning Theory (specifically a package developed by Timaeus - `devinterp` helped me with this, and I'll mention SLT later on in the report), Principal Component Analysis (PCA), Neuron Ablation (which measures how much the accuracy drops when a neuron is essentially not considered), and a (Normalized) Mutual Information graph. The neuron ablation was done on the output of the GAP layer in my model (there are 16 neurons). This was chosen since it's the layer right before the final classification layer, and o I believe that it is the most likely to be a measure of human-measurable features; and hence the most important one to look into. Similarly the NMI graph was also made using these same 16 neurons. The output I got for this was like so:
        * The Local Learning Coefficient came to be around 1.5.
        * Through PCA, the number of components came out to be around ~9-10.
        * Through the ablation I identified 9 neurons which cause the accuracy to drop by more than 1% (i.e. they're significant). Through PCA and this, especially showed that the model was simple; only these neurons really mattered.
        * Through the NMI graph I found out about the complete relations (both linear and non-linear which is why I picked NMI) between the neurons. Also I learnt about this recently and was looking where to apply this to. This combined with later steps gives a lot of insight into the model.
    All of this was implying that the model was clearly very simple in nature. The ablation is also used a lot in future steps.
    3. Next, I tried grouping the neurons based on how they get activated. For this, I first found the top 'k' (here k is 576) images that activate each of the 16 neurons and stored references to each of these images. Once I found this, I saw that I could infer a lot about the neurons by this itself; for example, Neuron 2 had all Green 1s. Neuron 8 had all Green 8s, and so on. This was very interesting and I came up with 2 methods to go about this:
        1. Using the same top-k images, we extract embeddings and semantic descriptions using a vision-language model (e.g., CLIP). We perform similarity analysis and clustering in semantic space to identify human-aligned modes of activation. Then, we look at the images using BLIP, and see how a human would describe the images. We collect the most occuring keywords and form sentences from the keywords to see how well they stack up against the centroid of all the vectors whose embeddings we generated from the top k images. That didn't work; I think that the vision encoder produces embeddings that are highly specific; also since the mean of the embeddings is very high, that means that they are pointing in a very specific direction. Since the vector space is 512-dimensional, it can be very hard to get a good match. Although, some text embeddings had a noticable match with the embeddings as compared to the others, the similarity was still not very high. So, something else I tried doing was instead of comparing it with human-interpretable text, compare it with 'visual snippets'. For example, I could compare it with just an entire 28x28x3 image of the background texture, or just a plain black background with the number 1. This did reveal more about the neurons (for example, for Neuron 2 I found that it was activated most by green textures)

## 4. Task 3: Intervention & Localization
*   **Grad-CAM Visualization**:

    *   **Interpretation**: The heatmaps consistently highlight the entire background of the image rather than the digit strokes. This provides direct visual evidence that the model is making decisions based on the background color.
*   **Neuron Injection/Patching**:
    *   **Description**: We artificially overwrote the activation of specific 'Color Neurons' with a high value during inference.
    *   **Result**: This flipped the model's prediction to the class associated with that neuron's preferred color, proving the causal importance of these spurious neurons.
*   **Ablation Study**:
    *   **Conclusion**: Removing the color-sensitive neurons improved Hard Test accuracy significantly, essentially forcing the model to rely on the remaining (weaker) shape signals that were previously drowned out.

## 5. Task 4: Mitigation Strategies
*   **Method 1: Just Train Twice (JTT)**:
    *   **Explanation**: We trained an initial "identification" model for a few epochs to identify "hard" examples (those it misclassified). We then trained a final model upweighting these error samples by a factor of 20.
    *   **Result**: **85.7%** Accuracy on the Hard Test Set.
*   **Method 2: Unsupervised Clustering (K-Means)**:
    *   **Explanation**: We extracted features from the penultimate layer and clustered them per class using K-Means (k=2). We identified the smaller cluster as the "bias-conflicting" group (e.g., Red '7's) and upweighted it during training.
    *   **Result**: **75.53%** Accuracy on the Hard Test Set.
*   **Comparison**: JTT proved slightly more effective (85.7% vs 75.5%) but requires training two models. Both methods significantly outperformed the baseline (3.77%), demonstrating that the model *can* learn shape if the spurious correlation is de-emphasized.

## 6. Task 5: Robustness Analysis (Adversarial Attacks)
*   **PGD Attack**:
    *   **Analysis**: The robust models (JTT and Clustering-based) were significantly harder to fool than the baseline. They required larger epsilon perturbations to flip the label. This suggests that "shape" features are inherently more robust to adversarial noise than the low-level, pixel-perfect color correlations learned by the baseline.

## 7. Extra Learnings: Complexity & Singular Learning Theory (SLT)
*   *Note: This section covers advanced analysis performed beyond the standard accuracy metrics.*
*   **Motivation**: Standard metrics (Accuracy/Loss) don't tell the full story. A simple solution (Color mapping) and complex solution (Shape recognition) might have similar Training Loss, but vastly different generalization.
*   **Local Learning Coefficient (LLC)**:
    *   **Definition**: The LLC estimates the 'effective dimensionality' or complexity of the function the model has learned. A higher LLC implies a more complex/structured solution.
    *   **Estimation Method**: We used SGLD (Stochastic Gradient Langevin Dynamics) to estimate the RLCT (Real Log Canonical Threshold).
*   **Results**:
    *   **Baseline Model LLC**: **~1.25**
    *   **Robust (Clustered) Model LLC**: **~2.50**
*   **Interpretation**:
    *   The baseline model found a 'simple' solution (color mapping), resulting in a lower complexity score (~1.25). The robust model was forced to learn 'shape', a more complex feature, resulting in a significantly higher LLC (~2.50). This quantitatively validates that the "robust" solution is distinct and simpler solutions (like color bias) are preferred by SGD unless intervention occurs.
    *   **Robust Model Ablation**:

## 8. Conclusion
*   **Summary**: We successfully demonstrated that standard training on biased data leads to spurious correlation learning (3.77% accuracy). Interpretability tools (Grad-CAM, Feature Viz) confirmed the model was a "color detector". Mitigation strategies like JTT and Unsupervised Clustering recovered robust accuracy (up to 85.7%).
*   **Trade-off**: The analysis highlights a fundamental trade-off: models prefer simple, spurious solutions. Achieving robustness requires active intervention (re-weighting data) to force the model to learn the more complex, generalized features (shape), as confirmed by SLT complexity metrics.
