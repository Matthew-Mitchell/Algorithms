# Algorithms

This repository contains custom algorithm implementations from current research papers which I could not find current implementations of.  

The first is RUSboost, which stands for Random Under Sampling Boosting. The algorithm helps to improve classification algorithms where there is a class imbalance problem. To counteract class imbalance, the algorithm intelligently undersamples the majority class. The algorithm starts by randomly selecting objservations from the majority class in order to create a balanced dataset to train an initial model on. From there, additional *weak* learners are added to create an ensemble (hence the name boost). The intrigue of the algorithm comes in how random under sampling is performed on these successive additions to the model. On each successive iteration, probabilities are assigned to each of the majority class observations proportional to the misclassification rate of that observation from the current panel of weak learners. As a result, observations which are prone to misclassification are apt to become part of the undersampled training set for the next iteration, and these additional weak learners should thereby improve results on these cases.