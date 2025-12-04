# Toddalia asiatica (L.) Lam.Region Classifier
This is a Toddalia asiatica origin classifier designed for small-sample Toddalia asiatica origin classification.

When using this tool, you need to download four files: region_classifier_Ul.py, region_classifier.py, Test Set.xlsx, Training Set.xlsx. After that, simply run region_classifier_Ul.py.
In the UI interface, the training set is imported by clicking the “Load Training Data” button. The dataset format requires samples to be arranged in rows and components in columns, presented as a two-dimensional array where each element represents the component content of a sample.

When you click “Train Model”, a window will pop up showing qualified models that meet the preset criteria. To predict test samples using the trained model, select an Excel file via the “Select Excel File” option and start the prediction by clicking “Start Prediction”.

The output results include the predicted origin, true label, prediction confidence, and correctness. Clicking the “Explain Prediction” button provides the basis for each sample's prediction, displaying three types of distances and their corresponding weights. The region with the smallest weighted distance is chosen as the final prediction result.
Users can view detailed information about the optimal model by clicking “Analyze Feature Contributions”, which contains five tabs: “PCA Components Scores”, “Principal Components Analysis”, “Component Feature Influence”, “Components Feature Contribution”, and “Summary”.

Additionally, the “Save Model” and “Load Model” buttons allow for quick initiation of subsequent analysis tasks.
