# CHE4230_Project
Advance Process Control Final Project
**Data Preprocessing**
The data processing was conducted based on prior statistical visualizations (scatter plots) and descriptive analyses (mean, median). These approaches were used to address missing values and outliers. After cleaning, the data was normalized.
**Unsupervised Methods**
Different combinations of dimensionality reduction and clustering techniques were used to generate labels for data classification. The Silhouette coefficient and Davies-Bouldin index were employed to qualitatively evaluate whuch methods achieved better clustering and classification results.
****Dimensionality Reduction: ****
  - PCA
  - TSNE
  - ISO
  - UMAP
  - Kernal PCA
****Clustering: ****
  - Kmeans
  - DBSCAN
  - HDBSCAN
**Supervise Methods**
After generating the labels, ANN and SVM models were trained, validate and tested to evaluate their performance. Hyperparameter optimization was applied to enhance the model's performance as much as possible.

After generating the labels, ANN and SVM models were trained, validated, and tested to evaluate their performance. Hyperparameter optimization GridSearchCV was conducted to find the best combination for the parameters: hidden layers sizes, Learning rate init, max iter, validation score, test score and solver.
