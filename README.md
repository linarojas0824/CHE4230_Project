# CHE4230_Project
Advance Process Control Final Project<br>
The Tennessee Eastman (TE) process created by the Eastman Chemical Co. has been widely used as a benchmark process for evaluating process supervision methods. This project focuses on software development and project planning, as well as data analysis, pattern recognition, and the detection and classification of faults. <br>
<br>
**Data Preprocessing**
The data processing was conducted based on prior statistical visualizations (scatter plots) and descriptive analyses (mean, median). These approaches were used to address missing values and outliers. After cleaning, the data was normalized.

**Unsupervised Methods** 
Different combinations of dimensionality reduction and clustering techniques were used to generate labels for data classification. The Silhouette coefficient and Davies-Bouldin index were employed to qualitatively evaluate whuch methods achieved better clustering and classification results.<br>

  **Dimensionality Reduction:** <br>
    - PCA <br>
    - TSNE <br>
    - ISO <br>
    - UMAP <br>
    - Kernal PCA.<br>

  **Clustering:** <br>
    - Kmeans <br>
    - DBSCAN <br>
    - HDBSCAN <br>

**Supervise Methods**
After generating the labels, ANN and SVM models were trained, validate and tested to evaluate their performance. Hyperparameter optimization was applied to enhance the model's performance as much as possible.<br> 
After generating the labels, ANN and SVM models were trained, validated, and tested to evaluate their performance. Hyperparameter optimization GridSearchCV was conducted to find the best combination for the parameters: hidden layers sizes, Learning rate init, max iter, validation score, test score and solver.
