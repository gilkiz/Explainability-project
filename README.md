# Explainability-project - Explaining ECG Disease Classifier
This repository includes an explainability framework for an image-based ECG disease classifier. 
The main notebook, **`explainability_pipeline.ipynb`**, requires a specific environment. 
Please download the pre-generated **`explainability_pipeline.html`**, and open it with a browser to see the results.


This project focuses on explaining the decisions of an ECG disease classifier using various interpretability techniques. We began by generating **saliency maps** and observed the importance of proper image cropping to avoid misleading results. **SmoothGrad** was applied to refine the saliency map, improving clarity without saving intermediate images.

For a case study, we analyzed an ECG image classified as positive for *Wolff-Parkinson-White disease*. Using **LIME**, we discovered potential flaws in the model's interpretability, as it highlighted irrelevant regions. A **counterfactual image** revealed that minimal changes could flip the model's prediction, indicating potential overfitting or lack of robustness.

We further explored **Captum's sliding window**, **Integrated Gradients**, **Noise Tunnel with SmoothGrad**, and **Gradient SHAP** to identify important pixels and regions. Each method provided insights into how the model processes both pixel-level and neighborhood information, revealing potential strengths and weaknesses in its decision-making process.
