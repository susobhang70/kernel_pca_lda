# Kernel PCA and Kernel LDA

Kernel PCA and Kernel LDA Implementation in Python using RBF Kernel, and using SVM to classify reduced dimensional data

## Datasets 
- <a href="https://archive.ics.uci.edu/ml/datasets/Arcene"> The Arcene dataset </a> 
- <a href="https://archive.ics.uci.edu/ml/datasets/Madelon"> The Madelon dataset </a> 

## Run

- `python KPCA_KLDA.py`
- `python SVM_KPCA_KLDA.py`

**Kernel Function used : RBF kernel**  

## Implementations

- `KPCA_KLDA.py` - This implements the kernel PCA and also the kernel LDA technique. Separate functions have been made for the same. The kernel used here is the `RBF` kernel. The kernel LDA is implemented from the <a href="https://en.wikipedia.org/wiki/Kernel_Fisher_discriminant_analysis">this source</a>

- `SVM_KPCA_KLDA.py` - This uses SVM from sklearn. It imports `KPCA_KLDA.py` file for the above mentioned functions, and reduces the data using these functions, and then uses SVM to classify data. Ther kernel used for SVM here is also RBF kernel.

For the above two files, the Madelon dataset can also be used. Modify the above files based on the info given above.  