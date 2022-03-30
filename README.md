# Feature-Importance
In Machine Learning, feature importance gives (usually just) a relative ranking of the predictive strength among the features of the model; useful for simplifying models and possibly improving generality. It helps us better understand the relationship between features & the target variable and features & features. When traning the model, some features may be highly correlated, which results in multicollinearity. After filtering out the correlated features, the model's performance may increase significantly. By calculating feature importance, you can determine which features is impacts the target variables the most. 

## Methods of Calculating Feature Importance

- Spearman's Rank Coefficient
- PCA
- mRmR
- Drop Column Importance
- Permutation Column Importance

Below is a graph showing the effectiveness of each method when using different models. x-axis represents number of features and y-axis represents MAE loss.

## Automatic feature selection algorithm

After calculating a score for each feature representing their importances. We want to implement an automated mechanism that selects the top k features automatically that gives the best validation error. The algorithm is shown below:
  - Step1: Calculate feature importance using permutation method
  - Step2: Calculate the Validation Loss using MAE with all features selected
  - Step3: Drop a feature that has the lowest importance score
  - Step4: Redo Step2 and Step4 until MAE loss increases more than 0.1
  - Step5: Return the final features selected

## Conclusion
Feature importance refers to techniques that assign a score to input features based on how useful they are at predicting a target variable. There are many types and sources of feature importance scores, and this report mainly introduces five of them. Among these five methods, the "permutation column" method seems to be most useful and generate the best results. This report also shows an algorithm that automatically select K features using permutation method. Other than that, when considering feature importances, standard deviation and p-values are useful to take into consideration.
