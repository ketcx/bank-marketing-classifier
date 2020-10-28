<h1 align='center'>Classification of clients of a bank's marketing campaign</h1>
<p align="center">Armando Medina</p>
<p align="center">(October, 2020)</p>

<p align="center">
  <img src="https://github.com/ketcx/bank-marketing-classifier/blob/master/data/diagram.png" width=600>
</p>

## Table of Contents

<details open>
<summary>Show/Hide</summary>
<br>

1. [Introduction](#1-introduction)
2. [Business Problem ](#2-business-problem)
3. [Data](#3-data)
4. [Methodology](#4-methodology)
5. [Results and Discussion](#5-results-and-discussion)
6. [Conclusion](#6-conclusion)
7. [Proof of cluster clean up](#7-proof-of-cluster-clean-up)
</details>

## 1. Introduction

<details>
<a name="#1-introduction"></a>
<summary>Show/Hide</summary>
<br>
  <p>This project is part of the Udacity Azure Machine Learning Nanodegree. In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model. This model is then compared to an Azure AutoML run.
  </p>
  <p>The specific project is based on analyzing the data that we have from the clients to determine if they will subscribe or not to the service offered, which is a term deposit.</p>
</details>

## 2. Business Problem

<details>
<a name="#2-business-problem"></a>
<summary>Show/Hide</summary>
<br>
<p>
  Customer acquisition is always a non-trivial problem in any company, regardless of the channel used to acquire customers, capture leads and convert them into customers of the company's products is a task that requires time and money. Therefore, companies would like to be able to predict if a given client will subscribe into a given product offered through a phone call. 
</p> 
<p>
Specifically, we explore a set of data related to direct marketing campaigns (phone calls) of a Portuguese banking institution. The problem we want to solve is to predict through the information if a client is going to subscribe or not a term deposit offered through a phone call.
</p>

</details>

## 3. Data

<details>
<a name="#3-data"></a>
<summary>Show/Hide</summary>
<br>
  <p>The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls.
  </p>
<p>
The data is related with direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe to a term deposit (variable y).
</p>

- For our work we will use the link offered by the Nanodegree that is located at:
  [https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv](https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv)

- A variant of this data set can be found in:
  [https://archive.ics.uci.edu/ml/datasets/Bank+Marketing](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)

<p align="center">
  <img src="https://github.com/ketcx/bank-marketing-classifier/blob/master/data/data1.png" width=600>
</p>
</details>

## 4. Methodology

<details>
<a name="#4-methodology"></a>
<summary>Show/Hide</summary>
<br>
  <p>In this project, our objective is to be able to predict if a client is going to subscribe or not to a long-term deposit. For this, we are going to use a scikit-learn model and with the help of the Python SDK we are going to optimize its hyperparameters through HyperDrive. Then, we are going to apply AutoML to the dataset. inally, we are going to compare the best model thrown from AutoML with the linear regression model optimized with the help of HypeDriver.
</p>
  <p>In a detailed way, we will follow the following steps:</p>

1. Inside the train.py file, we are going to load our dataset with the help of TabularDatasetFactory.

2. Then, we are going to clean our dataset for better handling, for this we are going to help the clean_data function (located in cleandata.py). With this function, using pandas, we are going to eliminate null values ​​if they exist, and transform some columns as "housing" for a better result of our model.

3. Once the ETL is passed, we are going to divide our data into training data and test data (or validation).

4. In this step, usingAzure ML and HyperDriver, we are going to optimize our model specifically in the parameters of Regularization Strength (C) and Max iterations (max_iter). Also, in our configuration file, we pass our early stopping policy and our estimator with our model.

5. Once HyperDriver finishes optimizing our model, we will register the best model and analyze the results of this compared to the others, for this we will help with accuracy.

6. After the optimization of our Linear Regression model with HyperDrive, we will implement AutoML to our dataset, for this we create an experiment passing it as parameters: our input data, our validation data, the type of task that in this case is the classification ( yes or no), the column that we want to predict, which in this case is "y", our metric that in order to compare with the previous model we will use "accuracy", we specify a timeout time and finally, in this case, we will specify two models that we do nott want AutoML to use in its search for the best model for our problem.

7. Once our experiment is finished, we will register the best model and compare it with our previous result.

8. When analyzing the Confusion Matrix of both models we observe that the model has biases due to the imbalance of our dataset. For this, we use the SMOTE technique applied in the "y" column and repeat the experiments already carried out.

<p>
About hyperparameter tuning, the logistic regression used does not really have any critical hyperparameters to fit.
</p><p>
That said, for the experiment we use the parameter C controls the intensity of the penalty, which for this type of algorithm can be effective. The other parameter we used was the max_iter, which is the number of iterations that the logistic regression classifier solver can go through before stopping. The objective is to arrive at a "stable" solution for the parameters of the logistic regression model. With this we can measure how many interactions are necessary to obtain a good precision in a reasonable time. If your max_iter is too low, you may not reach an optimal solution. If your value is too high, you can essentially wait forever for a low-precision solution.</p>
</details>

## 5. Results and Discussion

<details>
<a name="#5-results-and-discussion"></a>
<summary>Show/Hide</summary>
<br>
<p>
  As a result, we can say that there is not much difference between the two final models, although the best model produced by AutoML predicts slightly better.Since the difference is not significant, it must be validated how both models generalize. 
</p>
<p>
For our model resulting from the optimization of parameters with HyperDrive, we have that the four results offered similar performances with accuracy metrics of 91% and a training execution time between 1:34 - 1:41.</p>
<p align="center">
  <img src="https://github.com/ketcx/bank-marketing-classifier/blob/master/data/h001.png" width=600>
</p>
<p align="center">
  <img src="https://github.com/ketcx/bank-marketing-classifier/blob/master/data/h002.png" width=600>
</p>
<p align="center">
  <img src="https://github.com/ketcx/bank-marketing-classifier/blob/master/data/h003.png" width=600>
</p>
<p>Regarding the early termination policy, it was defined based on slack criteria and a frequency for evaluation. This early termination policy prevents experiments from running for a long time and using resources unnecessarily.
</p>
<p>In machine learning, a hyperparameter is a parameter whose value is used to control the learning process.</p>
<p>
In the case, the hyperparameters of the best model were the following:
</p>

- <strong>max_iter=100</strong>| Maximum number of iterations of the optimization algorithm.

- <strong>C= 73.5313</strong> | Each of the values in C describes the inverse of regularization strength. Like in support vector machines, smaller values specify stronger regularization.

<p align="center">
  <img src="https://github.com/ketcx/bank-marketing-classifier/blob/master/data/h004.png" width=600>
</p>

<p>The columns that most influence the prediction of this model:</p>

1. <strong>Last contact duration:</strong> This attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.

2. <strong>Number of Employees – Quarterly indicator:</strong> Number of employed persons for a quarter.

3. <strong>Employment variation rate:</strong> It refers to cyclical employment variation.

4. <strong>Three Month euribor:</strong> Euribor is short for Euro Interbank Offered Rate. The Euribor rates are based on the interest rates at which a panel of European banks borrow funds from one another.

5. <strong>Consumer price index:</strong> The Consumer Price Index (CPI) is a measure of the average change over time in the prices paid by urban consumers for a market basket of consumer goods and services.

<p align="center">
  <img src="https://github.com/ketcx/bank-marketing-classifier/blob/master/data/h005.png" width=600>
</p>

<p>Regarding the performance of our "best_run" we can see that the model manages to classify the "no" very well but still has problems to classify the "yes" correctly.</p>

</p>In detail:</p>

- 98% of the "no" were classified correctly.
- 2% of the "no" were classified as "yes" incorrectly.
- 40% of the "yes" were classified correctly.
- 60% of the "yes" were classified as "no" incorrectly.

<p align="center">
  <img src="https://github.com/ketcx/bank-marketing-classifier/blob/master/data/h007.png" width=600>
</p>

<p>Now we are going to analyze our models generated by AutoML.
</p>

<p>To begin with, one of the things that called our attention was that AutoML warned us that the dataset had a balance problem which increased the probability of bias, we will see it in detail later.</p>

<p align="center">
  <img src="https://github.com/ketcx/bank-marketing-classifier/blob/master/data/a001.png" width=600>
</p>

<p>Automated machine learning, also referred to as automated ML or AutoML, is the process of automating the time consuming, iterative tasks of machine learning model development.</p>
<p>AutoML allows you to train, evaluate, improve, and deploy models based on your data, which allows us to test and discard hundreds of models in the time it would take to test one.</p>
<p>In this particular case AutoML tested the dataset with around 32 different models.</p>

<p align="center">
  <img src="https://github.com/ketcx/bank-marketing-classifier/blob/master/data/a002.png" width=600>
</p>

<p>For our model resulting from implementing AutoML to our dataset, the precision metrics were between 72% and 91% with an execution time between 0:29 seconds and 0:45 seconds</p>
<p>The best model was the VottingEsemble followed by the MaxAbsScaler, LightBGM. However both a 91% accuracy similar to our HyperDrive optimized model.
</p><p>
Regarding the AutoML result, it is consistent that the best model was a voting ensemble model. A voting ensemble involves summing the predictions made by classification models or averaging the predictions made by regression models.
</p>

<p align="center">
  <img src="https://github.com/ketcx/bank-marketing-classifier/blob/master/data/a003.png" width=600>
</p>

<p>The hyperparameters used by AuotML in the best model were the following:</p>

- <strong>max_iter=100:</strong> Maximum number of iterations of the optimization algorithm

- <strong>Cs= 10:</strong> Each of the values in C describes the inverse of regularization strength. Like in support vector machines, smaller values specify stronger regularization.

- <strong>tol=0.0001:</strong> Tolerance for stopping criteria.

- <strong>solver=’lbfgs’:</strong> Algorithm to use in the optimization problem.

- <strong>penality=’l2’:</strong> Used to specify the norm used in the penalization. The ‘newton-cg’, ‘sag’ and ‘lbfgs’ solvers support only l2 penalties. ‘elasticnet’ is only supported by the ‘saga’ solver.

- <strong>intercept_scaling=1.0:</strong> Useful only when the solver ‘liblinear’ is used and self.fit_intercept is set to True. In this case, x becomes [x, self.intercept_scaling], i.e. a “synthetic” feature with constant value equal to intercept_scaling is appended to the instance vector.

<p>The columns that most influence the prediction of this model:</p>

1. <strong>Employment variation rate:</strong> Is referring to cyclical employment variation.

2. <strong>Last contact duration:</strong> This attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.

3. <strong>Number of Employees – Quarterly indicator:</strong> Number of employed persons for a quarter.

4. <strong>Month:</strong> Last contact month of year.

5. <strong>Contact Cellular:</strong> If the type of communication was through a cell phone.
<p align="center">
  <img src="https://github.com/ketcx/bank-marketing-classifier/blob/master/data/a004.png" width=600>
</p>

<p>Regarding the performance of the best model selected by AutoML, we can see that the classification of the true "no" is improved and the yes is slightly better.</p>

<p>In detail:</p>

- 95% of the "no" were classified correctly.
- 4% of the "no" were classified as "yes" incorrectly.
- 60% of the "yes" were classified correctly.
- 39% of the "yes" were classified as "no" incorrectly.
<p align="center">
  <img src="https://github.com/ketcx/bank-marketing-classifier/blob/master/data/a005.png" width=600>
</p>

<p><strong>The benefits of the chosen parameter sampler</strong></p>
<p>Azure Machine Learning supports the following parameter sampling methods:</p>

- Random sampling: supports discrete and continuous hyperparameters. It supports early termination of low-performance runs.

- Grid sampling: supports discrete hyperparameters. Use grid sampling if you can budget to exhaustively search over the search space. Supports early termination of low-performance runs.

- Bayesian sampling: only supports choice, uniform, and quniform distributions over the search space. Bayesian sampling is recommended if you have enough budget to explore the hyperparameter space.

<p>Our selection of RandomSampling is motivated because Regularization Strength is a continuous hyperparameter. In other words, random sampling allowed my parameters to be initialized with both discrete and continuous values, and it also allowed for early political termination. This choice gave us an appropriate cost/benefit result.
</p>

<p>
As a basis for future work, you can read more about the difference between Grind Sampling and Random Sampling in James Bergstra & Yoshua Bengio's article: 
  <i>Random Search for Hyper-Parameter Optimization: </i> https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf
</p>

<p><strong>Handle imbalanced data</strong></p>
<p>The variable y is extremely unbalanced, this causes bias, this can be seen in the confusion matrix.</p>

<p><strong>Handle imbalanced data</strong></p>
<p>The variable y is extremely unbalanced, this causes bias, this can be seen in the confusion matrix.</p>
<p align="center">
  <img src="https://github.com/ketcx/bank-marketing-classifier/blob/master/data/balanced001.png" width=600>
</p>
<p>In the handle-imbalanced-data.ipynb notebook included in this project, you can see how the problem is corrected and the dataset is created through the Synthetic Minority Oversampling Technique, or SMOTE for short.</p>

<p align="center">
  <img src="https://github.com/ketcx/bank-marketing-classifier/blob/master/data/balanced002.png" width=600>
</p>
<p align="center">
  <img src="https://github.com/ketcx/bank-marketing-classifier/blob/master/data/balanced003.png" width=600>
</p>

<p>Once the training data was balanced (we left the unbalanced validation date) we ran our experiments for both our experiments and the results were the following:</p>

- For the best model with hyperparameters optimized with HyperDrive (C:84.379 & MAX_ITER 100):

  - 96% of the "no" were classified correctly.
  - 4% of the "no" were classified as "yes" incorrectly.
  - 49.51% of the "yes" were classified correctly.
  - 50.49% of the "yes" were classified as "no" incorrectly.
  <p align="center">
    <img src="https://github.com/ketcx/bank-marketing-classifier/blob/master/data/CM_HB.png" width=600>
  </p>

- For the best AutoML model:

  - 98% of the "no" were classified correctly.
  - 2% of the "no" were classified as "yes" incorrectly.
  - 66% of the "yes" were classified correctly.
  - 34% of the "yes" were classified as "no" incorrectly.

<p align="center">
    <img src="https://github.com/ketcx/bank-marketing-classifier/blob/master/data/CM_A.png" width=600>
  </p>
</details>

## 6. Conclusion

<details>
<a name="#6-conclusion"></a>
<summary>Show/Hide</summary>
<br>
  <p>The end result of optimizing the hyperparameters with HyperDrive and generating a model with AutoML is quite similar. During the experiments carried out with the dataset, the models gave a prediction of 91% accuracy.</p>

<p>However, this 91% is cheating, mainly because our dataset is imbalanced which produces bias. Realizing this, we applied the SMOTE technique to the column of the customer's response and the results of the models when predicting an if improvement. In the case of the best model, after optimization with HyperDrive, it went from classifying 40% of the "yes" correctly to classifying almost 50% correctly./<p>

<p>However, our dataset is still slightly balanced, especially in two that affect the prediction.</p>

<p>For future work and to obtain better results, two things must be done primarily:</p>

1. Eliminate the last contact duration column in order to bring the models closer to a real-world problem.

2. Correct the balance problem: In every ML project, data management usually represents more than 80% of the work, in this case, there is evidence that more work is needed in the data set, mainly to correct the imbalance. Unbalanced data can lead to a falsely perceived positive effect of a model's precision because the input data is biased towards one class.

3. For future work, it would be interesting to apply Hypedriver to the five best AutoML result models, in addition, it would also be interesting to test Hypedriver with other parameters such as "tol", "solver" and "penalty" that AutoML used during the selection of your model.

</details>

## 7. Proof of cluster clean up

<details>
<a name="#7-proof-of-cluster-clean-up"></a>
<summary>Show/Hide</summary>
<br>
  <p align="center">
    <img src="https://github.com/ketcx/bank-marketing-classifier/blob/master/data/cleanup001.png" width=600>
  </p>
  <p align="center">
    <img src="https://github.com/ketcx/bank-marketing-classifier/blob/master/data/cleanup002.png" width=600>
  </p>
</details>
