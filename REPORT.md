# CIFAR-10 Pure Linear Regression Report

## Experiment Setup

- Dataset: CIFAR-10
- Training subset: 5,000 images
- Test subset: 1,000 images
- Input representation: flattened raw pixels (`3072` features)
- Preprocessing: divide by `255`, standardize with training mean/std, add bias term
- Model: one-vs-rest pure linear regression
- Cross-validation: 5 folds on the training subset

## 5-Fold Cross-Validation Results

| Fold | Validation Accuracy |
| --- | --- |
| 1 | 14.90% |
| 2 | 14.50% |
| 3 | 14.20% |
| 4 | 15.20% |
| 5 | 16.40% |

Mean cross-validation result:

- mean validation accuracy = `15.04%`

## Final Test Result

- test accuracy = `19.80%`

## Confusion Matrix For Best Model

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

Class mapping:

- `0`: airplane
- `1`: automobile
- `2`: bird
- `3`: cat
- `4`: deer
- `5`: dog
- `6`: frog
- `7`: horse
- `8`: ship
- `9`: truck

## Short Analysis

The overall accuracy is limited because pure linear regression uses a linear decision boundary on raw pixels. CIFAR-10 images contain large variations in pose, color, lighting, and background, which raw-pixel linear models do not represent well.

The confusion matrix shows that visually related classes are mixed often:

- `cat`, `dog`, and `deer`
- `automobile` and `truck`
- `airplane` and `ship`

This confirms that pure linear regression is simpler than KNN at prediction time, but it is still not strong enough for image classification when only raw pixels are used.

## Plot

- `linear_regression_5fold_cv.png`: graph of fold accuracy in 5-fold validation
