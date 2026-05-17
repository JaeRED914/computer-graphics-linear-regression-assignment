# CIFAR-10 Pure Linear Classification Report

## Requirement Check

1. Data split:
   - official CIFAR-10 training split used for model development
   - 5,000-image training subset further split into 5 folds for train/validation
   - official CIFAR-10 test split used separately for final testing
2. 5-fold cross-validation:
   - yes
3. Metrics reported:
   - accuracy
   - precision
   - recall
   - F1-score
   - confusion matrix
4. Model form:
   - pure linear model with bias term, `Wx + b`
   - used as a classifier through class-score comparison
5. Hyperparameter graph:
   - not applicable in this pure linear classification version because no regularization hyperparameter was used
   - instead, a fold-wise validation accuracy graph is provided

## Dataset Structure

The CIFAR-10 python version contains:

- `data_batch_1` to `data_batch_5`: training data
- `test_batch`: test data
- `batches.meta`: class names

Each sample is:

- a color image of size `32 x 32 x 3`
- stored as `3072` raw pixel values after flattening
- labeled as one of 10 classes:
  `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`

## Experiment Setup

- Training subset used for CV: 5,000 images
- Validation method: 5-fold cross-validation
- Test subset: 1,000 images from the official CIFAR-10 test batch
- Input representation: flattened raw pixels (`3072` features)
- Preprocessing: divide by `255`, standardize with training mean/std, add bias term
- Model: one-vs-rest pure linear classifier using a linear regression formulation

## 5-Fold Cross-Validation Results

| Fold | Validation Accuracy |
| --- | --- |
| 1 | 14.90% |
| 2 | 14.50% |
| 3 | 14.20% |
| 4 | 15.20% |
| 5 | 16.40% |

Mean cross-validation accuracy:

- `15.04%`

## Final Test Metrics

- Accuracy: `19.80%`
- Macro Precision: `19.65%`
- Macro Recall: `19.59%`
- Macro F1-score: `19.55%`

## Confusion Matrix

```text
true\pred     0     1     2     3     4     5     6     7     8     9
        0    18     4    11    10     7     8     5    12    19     9
        1     8    15     9     7     9     2     7     7    10    15
        2    12     4    16    10    12    10     7     9    12     8
        3     6     6     9    22     8    15    16     8     9     4
        4     6     5    10    15    15     7     9     6     6    11
        5     6     3     7    12     9    14     7    12    11     5
        6     8     4    15    16    12    13    25     6     7     6
        7     6     7     8     6    15     9    11    19    10    11
        8    13    13     8     5     5     3    11     8    30    10
        9    13    11     5    11     7     5     5    11    17    24
```

## Per-Class Precision, Recall, and F1-score

| Class | Precision | Recall | F1-score |
| --- | --- | --- | --- |
| airplane | 18.75% | 17.48% | 18.09% |
| automobile | 20.83% | 16.85% | 18.63% |
| bird | 16.33% | 16.00% | 16.16% |
| cat | 19.30% | 21.36% | 20.28% |
| deer | 15.15% | 16.67% | 15.87% |
| dog | 16.28% | 16.28% | 16.28% |
| frog | 24.27% | 22.32% | 23.26% |
| horse | 19.39% | 18.63% | 19.00% |
| ship | 22.90% | 28.30% | 25.32% |
| truck | 23.30% | 22.02% | 22.64% |

## Saved Figures

- `linear_classification_5fold_cv.png`: fold-wise validation accuracy
- `linear_classification_confusion_matrix.png`: confusion matrix heatmap
- `linear_classification_class_metrics.png`: class-wise precision/recall/F1-score bar chart

## Short Interpretation For Presentation

- The model used only raw pixels and a linear score function, so performance is limited.
- Validation accuracy stayed around `15%`, and final test accuracy was `19.80%`.
- `frog`, `ship`, and `truck` were relatively better recognized than classes such as `bird` or `deer`.
- The confusion matrix shows many errors between visually similar classes.
- This is consistent with the weakness of pure linear classification on image classification.

## Submission Judgment

If the professor mainly wanted "do the previous CIFAR assignment again, but this time with linear regression," this version is defensible because:

- it uses CIFAR-10 correctly
- it separates train/validation/test usage properly
- it uses 5-fold cross-validation
- it reports the main evaluation metrics
- it uses a pure linear model with `Wx + b`
- it clearly states that the final task is classification
- it includes presentation-friendly graphs

The only caveat is that a pure linear classifier has no natural K-like hyperparameter. So for this version, the validation graph is over folds rather than over a hyperparameter axis.
