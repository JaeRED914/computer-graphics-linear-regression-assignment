# CIFAR-10 Pure Linear Regression Assignment

This project adapts the old CIFAR-10 KNN assignment into a pure linear regression classifier assignment.

## Goal

Implement a pure linear regression based image classifier from scratch on the CIFAR-10 dataset using raw pixel features. Use 5-fold cross-validation on the training subset to estimate validation performance, report validation accuracy, test accuracy, and a confusion matrix, and briefly analyze the limitations of linear regression for image classification.

## Assignment Flow

1. Load CIFAR-10 from the local `cifar-10-batches-py` folder.
2. Use a small subset first:
   - 5,000 training images
   - 1,000 test images
3. Split the 5,000 training images into 5 folds.
4. Flatten each image from `32 x 32 x 3` into `3072` raw-pixel features.
5. Normalize each fold using only the corresponding training-fold statistics.
6. Add a bias term.
7. Implement one-vs-rest pure linear regression manually.
8. Since pure linear regression has no regularization hyperparameter here, use 5-fold cross-validation to estimate validation accuracy.
9. Train on 4 folds and validate on the remaining fold.
10. Repeat this process 5 times and compute the mean cross-validation accuracy.
11. Plot:
   - X-axis: fold number
   - Y-axis: validation accuracy
12. Retrain on the full training subset and evaluate on the test subset.
13. Show the confusion matrix for the final model.

## Files

- `linear_regression_cifar10.py`: main implementation
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
- the confusion matrix of the best model
- CIFAR-10 class index mapping

It also saves:

- `linear_regression_5fold_cv.png`: cross-validation plot

## Brief Analysis

Pure linear regression is much simpler and faster at inference than KNN, but it is still a weak image classifier when it only uses raw pixels:

- it assumes a mostly linear decision boundary
- it does not capture local image structure well
- it is sensitive to background, pose, and color variation
- semantically similar classes such as `cat` and `dog` or `automobile` and `truck` are often confused

If you want stronger performance, the next step is a softmax classifier or a convolutional neural network.
