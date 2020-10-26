from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from cleandata import clean_data

from interpret.ext.blackbox import TabularExplainer
from azureml.interpret import ExplanationClient

OUTPUT_DIR = './outputs/'
os.makedirs(OUTPUT_DIR, exist_ok=True)
run = Run.get_context()
client = ExplanationClient.from_run(run)

data_train = pd.read_csv('data/data_balanced.csv')  
data_test = pd.read_csv('data/data_validation.csv')  

y_train = data_train["y"]
data_train.drop("y", inplace=True, axis=1)
data_train.drop("Unnamed: 0", inplace=True, axis=1)
x_train = data_train

y_test = data_test["y"]
data_test.drop("y", inplace=True, axis=1)
data_test.drop("Unnamed: 0", inplace=True, axis=1)
x_test = data_test

feature_names = list(x_train.columns)

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))
    
    os.makedirs('outputs', exist_ok=True)
    # files saved in the "outputs" folder are automatically uploaded into run history
    joblib.dump(model, 'outputs/model.joblib')
    
    #model_name
    model_file_name = 'model.joblib'
    
    # register the model
    run.upload_file('original_model.pkl', os.path.join('./outputs/', model_file_name))
    original_model = run.register_model(model_name='model_explain',model_path='original_model.pkl')

    # Explain predictions on your local machine
    tabular_explainer = TabularExplainer(model, x_train, features=feature_names)
    global_explanation = tabular_explainer.explain_global(x_test)

    # The explanation can then be downloaded on any compute
    comment = 'Global explanation on regression model trained on bank marketing campaing dataset'
    client.upload_model_explanation(global_explanation, comment=comment, model_id=original_model.id)    

if __name__ == '__main__':
    main()
