# Optimizing an ML Pipeline in Azure


  ____              _        __  __            _        _   _                                 
 |  _ \            | |      |  \/  |          | |      | | (_)                                
 | |_) | __ _ _ __ | | __   | \  / | __ _ _ __| | _____| |_ _ _ __   __ _                     
 |  _ < / _` | '_ \| |/ /   | |\/| |/ _` | '__| |/ / _ \ __| | '_ \ / _` |                    
 | |_) | (_| | | | |   <    | |  | | (_| | |  |   <  __/ |_| | | | | (_| |                    
 |____/_\__,_|_| |_|_|\_\   |_|  |_|\__,_|_|  |_|\_\___|\__|_|_| |_|\__, |  _           _     
  / ____|                          (_)                    /\         __/ | | |         (_)    
 | |     __ _ _ __ ___  _ __   __ _ _  __ _ _ __         /  \   _ __|___/ _| |_   _ ___ _ ___ 
 | |    / _` | '_ ` _ \| '_ \ / _` | |/ _` | '_ \       / /\ \ | '_ \ / _` | | | | / __| / __|
 | |___| (_| | | | | | | |_) | (_| | | (_| | | | |     / ____ \| | | | (_| | | |_| \__ \ \__ \
  \_____\__,_|_| |_| |_| .__/ \__,_|_|\__, |_| |_|    /_/    \_\_| |_|\__,_|_|\__, |___/_|___/
                       | |             __/ |                                   __/ |          
                       |_|            |___/                                   |___/           


## Overview

This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary

**This project is based on the analysis of the marketing data of a banking institution.**

**In this project we work on the construction of a classification model to correctly analyze the behavior of the client through data such as: job, education, day of week, etc. As a result of this work, the bank could then use the same model to predict customer responses to future marketing initiatives.**

_The best performing model was VotingEsemble, even superior to the scikit-learn linear regression model with the hyperparameters tuning through Hyperdrive._

## Scikit-learn Pipeline

**A pipeline could be defined as the intermediate steps of the pipeline must implement fitting and transformation methods and the final estimator only needs to implement fitting.**

**For this project, select some sample parameters, data cleaning, create a termination policy, divide the data into training data and validation data.**

**The parameters that were chosen were Regularization Strength and Max iterations. Regularization Strength is Inverse of regularization strength. Smaller values cause stronger regularization and Max iterations is the maximum number of iterations to converge**

**Regarding the early termination policy, an early termination policy was defined based on slack criteria and a frequency for evaluation. This early termination policy prevents experiments from running for a long time and using resources unnecessarily.**

## AutoML

**In the second step, AutoML was in charge of generating the parameters and selecting the model. This was an ensemble voting model, which is a set machine learning model that combines the predictions of many other models.**

## Pipeline comparison

**For this classification problem and this dataset, in particular, the difference between AutoML and Sckit-learn are insignificant in the result, however, it can be seen that the use of AutoML allows to have a faster idea of which model works better and then this model is You can apply hyperparameter tuning to it with the help of Hyperdrive.**

## Future work

**For future work, it would be interesting to apply Hypedriver to the five best AutoML result models, in addition, it would also be interesting to test Hypedriver with other parameters such as "solver" and "penalty".**
