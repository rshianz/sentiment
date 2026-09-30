# sentiment
main.py is for plt show of data and train them (phase 1 & 2)

predict.py is for when you have done training and 'best_model_state' file is availible and you can write sentences and test the model (part of phase 4)



````markdown
# RoBERTa — Test Set Performance Report

**Overall Accuracy: 0.8283**

## Classification Report

| Class            | Precision | Recall | F1-Score | Support |
|------------------|-----------|--------|----------|---------|
| negative (0)     | 0.81      | 0.69   | 0.75     | 81      |
| neutral (1)      | 0.60      | 0.75   | 0.67     | 63      |
| positive (2)     | 0.92      | 0.90   | 0.91     | 217     |
| **accuracy**     | —         | —      | **0.83** | 361     |
| **macro avg**    | 0.78      | 0.78   | 0.77     | 361     |
| **weighted avg** | 0.84      | 0.83   | 0.83     | 361     |

## Confusion Matrix

|                      | Pred: negative | Pred: neutral | Pred: positive |
|----------------------|:--------------:|:-------------:|:--------------:|
| **Actual: negative** | 56             | 15            | 10             |
| **Actual: neutral**  | 8              | 47            | 8              |
| **Actual: positive** | 5              | 16            | 196            |
````
