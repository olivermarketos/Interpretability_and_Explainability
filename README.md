# ML4HC_Interpretability_and_Explainatility

## Part 1

### Q1: Exploratory Data Analysis
- The `p1/1_data_exploration.ipynb` file contains all the necessary code for this section.
- It uses the raw data (`train_val_split.csv` and `test_split.csv`) in the `p1/data` folder and also saves processed data into the same folder (`train_val.parquet` and `test.parquet`).
- The file contains code for visualizing the feature distributions and processing the data.

### Q2: Logistic Lasso Regression
- The code for the section is in the `p1/2_lasso_regression.ipynb` file.
- It contains all the necessary functions but uses the previously processed data in the `p1/data` folder (`train_val.parquet`, `test.parquet`).
> **note**: The part of the code that saves the model and the results on the test set is commented out.

### Q3: Multi-Layer Perceptrons
- The `p1/3_multi-layer-perceptrons.ipynb` file contains all the necessary functions and classes for this part.
- It also uses the processed data files in the `p1/data` folder (`train_val.parquet`, `test.parquet`).
> **note**: The part of the code that saves the model and the results on the test set is commented out.

### Q4: Neural Additive Models
- The same way as previously, the `p1/4_neural_additive_models.ipynb` file contains all the necessary functions and classes for this part.
- It also uses the processed data files in the `p1/data` folder (`train_val.parquet`, `test.parquet`).
> **note**: The part of the code that saves the model and the results on the test set is, again, commented out.
