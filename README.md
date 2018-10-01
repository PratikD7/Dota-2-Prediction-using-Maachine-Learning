# Dota 2 Prediction using Machine Learning.

Data for this project is collected using [Valve's Steam API!](https://developer.valvesoftware.com/wiki/Steam_Web_API) and is parsed into csv files.

There are two types of techniques used to classify this data:
* Supervised Learning using SVM with RBF Kernel
* Unsupervised Learning using Gaussian Mixture Models

The accuracy for SVM model is close to 94% whereas FOR GMM it is only 60%.
It made me conclude that this data is too complex for clustering algorithms to find relevant patterns.
