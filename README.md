# mutual_crossattetion_PDclustering
![figure1](https://github.com/user-attachments/assets/034acec0-54f6-46ff-9830-f26858f939dc)
The framework comprises four main stages: a, Extraction of PSD features from resting-state EEG data using a convolutional neural network trained to predict MDS-UPDRS-III scores. b, Extraction of gait features from motion assessments during single-task and dual-task walking and turning paradigms. c, Fusion of EEG PSD features with gait features (separately for single-task and dual-task conditions) using a mutual cross-attention (MCA) mechanism. d, Application of K-means clustering to the fused feature space (visualized using t-SNE) to identify distinct patient subtypes.

we provided a toy dataset.
Load the toy_dataset1.npy and toy_dataset2.npy files.
Pass them through the CrossAttention module.
Perform K-Means clustering on the attention output.
Generate a t-SNE plot visualizing the clusters.
