# CIFAR-10 Pure Linear Regression Assignment

This project adapts the old CIFAR-10 KNN assignment into a pure linear regression classifier assignment.

## Goal

Implement a pure linear regression based image classifier from scratch on the CIFAR-10 dataset using raw pixel features. Use 5-fold cross-validation on the training subset to estimate validation performance, report accuracy, precision, recall, F1-score, and a confusion matrix, and briefly analyze the limitations of linear regression for image classification.

## Dataset Structure

The local dataset folder is `cifar-10-batches-py/`.

- `data_batch_1` to `data_batch_5`: CIFAR-10 training data batches
- `test_batch`: CIFAR-10 official test data
- `batches.meta`: class name metadata
- `readme.html`: dataset description

CIFAR-10 contains:

- 10 object classes
- 50,000 training images total
- 10,000 test images total
- each image is `32 x 32 x 3`
- each flattened image has `3072` raw-pixel features

This project uses:

- 5,000 images from the official training split for model development
- 5-fold cross-validation inside those 5,000 images
- 1,000 images from the official test split for final evaluation

So the workflow is:

- `train/validation`: created from the 5,000-image training subset using 5-fold CV
- `test`: taken separately from the official CIFAR-10 test batch

## Assignment Flow

1. Load CIFAR-10 from the local `cifar-10-batches-py` folder.
2. Use a small subset first:
   - 5,000 training images
   - 1,000 test images
3. Split the 5,000 training images into 5 folds.
4. Flatten each image from `32 x 32 x 3` into `3072` raw-pixel features.
5. Normalize each fold using only the corresponding training-fold statistics.
6. Add a bias term so the model has the form `Wx + b`.
7. Implement one-vs-rest pure linear regression manually.
8. Since pure linear regression has no regularization hyperparameter here, use 5-fold cross-validation to estimate validation accuracy.
9. Train on 4 folds and validate on the remaining fold.
10. Repeat this process 5 times and compute the mean cross-validation accuracy.
11. Plot:
   - X-axis: fold number
   - Y-axis: validation accuracy
12. Retrain on the full training subset and evaluate on the test subset.
13. Report:
   - accuracy
   - precision
   - recall
   - F1-score
   - confusion matrix
14. Save presentation-friendly figures.

## Files

- `linear_regression_cifar10.py`: main implementation
- `REPORT.md`: experiment summary
- `linear_regression_5fold_cv.png`: 5-fold validation accuracy plot
- `linear_regression_confusion_matrix.png`: confusion matrix heatmap
- `linear_regression_class_metrics.png`: per-class precision/recall/F1 bar chart
- `cifar-10-batches-py/`: CIFAR-10 python batch files

## Environment

The script requires `numpy` and `matplotlib`.

Example setup with a virtual environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install numpy matplotlib
```

## Run

```bash
python3 linear_regression_cifar10.py
```

You can also change subset size:

```bash
python3 linear_regression_cifar10.py --num-train 3000 --num-test 500
```

## Expected Output

The script prints:

- validation accuracy for each fold
- mean cross-validation accuracy
- test accuracy
- macro precision
- macro recall
- macro F1-score
- confusion matrix
- per-class precision/recall/F1-score
- CIFAR-10 class index mapping

It also saves:

- `linear_regression_5fold_cv.png`
- `linear_regression_confusion_matrix.png`
- `linear_regression_class_metrics.png`

## Brief Analysis

Pure linear regression is much simpler and faster at inference than KNN, but it is still a weak image classifier when it only uses raw pixels:

- it assumes a mostly linear decision boundary
- it does not capture local image structure well
- it is sensitive to background, pose, and color variation
- semantically similar classes such as `cat` and `dog` or `automobile` and `truck` are often confused

If you want stronger performance, the next step is a softmax classifier or a convolutional neural network.
