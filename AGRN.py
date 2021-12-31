# Importing Libraries

import pathlib
import warnings
from sklearn.model_selection import train_test_split
from tqdm import notebook
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import *
from sklearn.preprocessing import StandardScaler
import socket
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import  ExtraTreesRegressor 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score,  average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import shap
import time
from joblib import Parallel, delayed
from sklearn.svm import LinearSVR
import random
import optuna
import mlflow
import sys
import os
from mlflow import log_metric, log_param, log_artifacts

#Igonre Warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

#Calculate regression results
def regressionResult(y_true, predicted):

    pearson = pearsonr(y_true, predicted)
    mae = mean_absolute_error(y_true, predicted)
    maepearson=1-mae+np.abs(pearson[0])
    
    return maepearson

#Objective function for Bayesian Search
def objective(trial,data,target):

    train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.25, random_state=42)

    # SVR will take the parameters
    param = {  
        'clf__epsilon': trial.suggest_loguniform('epsilon', 1e-5, 1e-1),
        'clf__C': trial.suggest_loguniform('C', 1e0, 1e4)
    }
    model = Pipeline([('scale', StandardScaler()),('clf', ML_instances["SVR"])])

    bst = model.fit(train_x, train_y)
    preds = bst.predict(valid_x)
    mae = regressionResult(valid_y, preds)
    return mae

#Importance Scores from SVR
def feature_importance(cc,column):
        print(cc,column)  
        flag=0

        noofsamples=samples
        rankThreshold=5
        df_sum=pd.DataFrame()       

        for k in range(noofsamples):

                rowfrac=random.uniform(0.2, 0.8)
                colfrac=random.uniform(0.2, 0.8)

                if fourth_line !="":
    
                    if(column in tf_list):
                        Ntrain_df=train_df[tf_list].copy()
                    else:
                        Ntrain_df=pd.concat([train_df[tf_list],train_df[column]],axis=1)
                else:
                    Ntrain_df=train_df.copy()

                Ntrain_df=Ntrain_df.sample(frac=rowfrac)
                Ntrain_df=pd.concat([Ntrain_df.drop(column, axis = 1).sample(frac=colfrac,axis="columns"),Ntrain_df[column]],axis=1)
                y_train=Ntrain_df[column].to_numpy()

                X_train = Ntrain_df.drop(column, axis = 1).to_numpy()           
                New_index=Ntrain_df.drop(column, axis = 1).columns+"_"+column
    
                optuna.logging.set_verbosity(optuna.logging.WARNING)

                study = optuna.create_study(direction="maximize")
  
                study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=notrial, timeout=40,show_progress_bar=False)
                # print(study.best_params)

                dic=study.best_params
                model = make_pipeline(StandardScaler(),LinearSVR(**dic))
                clf = model.fit(X_train, y_train)
                vals = np.abs(clf[1].coef_)
                coeff=pd.DataFrame(vals, index=New_index, columns=['feat_importance'])
                coeff.sort_values(by="feat_importance", inplace=True,ascending= False )
                coeff[0:rankThreshold]=1
                coeff[rankThreshold:len(coeff)]=0  
                if flag==0:                                              
                        df_sum=coeff.copy()
                        flag=1
                else:                      
                        df_sum = df_sum.add( coeff, fill_value=0)
        return df_sum
        
# %%
#Importance Scores from ETR and RFR
def feature_importance2(cc,column):
        print(cc,column)   

        y_train=train_df[column].to_numpy()

        if fourth_line !="":
            tftrain_df=train_df[tf_list]
        else:
            tftrain_df=train_df.copy()

        if   column in tftrain_df.columns:
                X_train = tftrain_df.drop(column, axis = 1).to_numpy()           
                New_index=tftrain_df.drop(column, axis = 1).columns+"_"+column
        else:
                X_train = tftrain_df.to_numpy()
                New_index=tftrain_df.columns+"_"+column

        
        model = Pipeline([('scale', StandardScaler()),('clf', ML_instances[cc]),])

        clf =model.fit(X_train, y_train)      

        explainer = shap.TreeExplainer(clf[1])
        shap_values = explainer.shap_values(X_train,check_additivity=False)     

        vals1 = np.abs(shap_values).mean(0)
        vals2 =clf[1].feature_importances_

        df= pd.concat([pd.DataFrame(vals1, index=New_index, columns=['feat_importance']) , pd.DataFrame(vals2, index=New_index, columns=['shap'])],axis=1)

        return df

#Calculate Classificiation Accuracy
def classificationResult(y, predicted,predicted_proba,Output_file,FileName,MethodName,flag=None):
   
    auc_score= round(roc_auc_score(y, predicted_proba), 4) 
    aucPR_score= round(average_precision_score(y, predicted_proba), 4) 

    if(flag==None):
        print("AUCROC (%),",round(auc_score, 3))
        print("AUCPR (%),",round(aucPR_score, 3))
        print("Average (%),",round((auc_score+aucPR_score)/2, 3))
 
    if(flag==None):
        print("AUCROC (%),",round(auc_score, 3),file=Output_file)
        print("AUCPR (%),",round(aucPR_score, 3),file=Output_file)
        print("Average (%),",round((auc_score+aucPR_score)/2, 3) ,file=Output_file)

        mlflow.start_run(run_name=FileName)
        mlflow.log_param("Method", MethodName)
        log_metric("AUPR", auc_score)
        log_metric("AUROC", aucPR_score)
        log_metric("Average", (auc_score+aucPR_score)/2)
        mlflow.end_run()

    return (auc_score+aucPR_score)/2

#Calculate the AUROC and AUPR based on the groudtruth data
def evalresults(groundtruth_path,result_path,Output_file,FileName):

    ground_truth=pd.read_csv(groundtruth_path.strip(),sep='\t',header=None)
    new_index=ground_truth[0]+"_"+ground_truth[1]
    ground_truth.index=new_index
    ground_truth=ground_truth.drop([0,1], axis = 1)
    ground_truth=ground_truth.sort_index()
    ground_truth=ground_truth.rename(columns={2: "GroundTruth"})
    ground_truth   

    ETR=pd.read_csv("./"+result_path+"/ETR.csv",index_col=0)
    ETR=ETR.sort_index()


    RFR=pd.read_csv("./"+result_path+"/RFR.csv",index_col=0)
    RFR=RFR.sort_index()
    

    if int(usr_input)==2:
        SVR=pd.read_csv("./"+result_path+"/SVR.csv",index_col=0)
        SVR=SVR.sort_index()

        threshold=0.5
        dic={"a":10,"b":5,"c":1}
        print("ShapBasedOnETR+ShapBasedOnRFR+SVR",file=Output_file)    
        ground_truth["Comb"]=(dic["a"]*RFR["shap_Proba"]+dic["b"]*ETR["shap_Proba"]+dic["c"]*SVR["SVRProba"])/(dic["a"]+dic["b"]+dic["c"])  
        ground_truth.loc[ground_truth["Comb"]>=threshold,"predict"]=1 
        ground_truth.loc[ground_truth["Comb"]<threshold,"predict"]=0 
        ground_truth["Comb"]=ground_truth["Comb"].fillna(0)
        classificationResult(ground_truth['GroundTruth'].to_numpy(),ground_truth["predict"].to_numpy(), ground_truth["Comb"].to_numpy(),Output_file,FileName,"ShapBasedOnETR+ShapBasedOnRFR+SVR")
        print("*****************************************************************************************\n")

    else:
        threshold=0.5
        print("ShapBasedOnETR+ShapBasedOnRFR",file=Output_file)    
        ground_truth["Comb"]=(RFR["shap_Proba"]+ETR["shap_Proba"])/2
        ground_truth.loc[ground_truth["Comb"]>=threshold,"predict"]=1 
        ground_truth.loc[ground_truth["Comb"]<threshold,"predict"]=0 
        ground_truth["Comb"]=ground_truth["Comb"].fillna(0)
        classificationResult(ground_truth['GroundTruth'].to_numpy(),ground_truth["predict"].to_numpy(), ground_truth["Comb"].to_numpy(),Output_file,FileName,"ShapBasedOnETR+ShapBasedOnRFR")
        print("*****************************************************************************************\n")

if __name__ == "__main__":

    # Output Directory 
    FileName="Output"
    # Input file
    inputfile='input.txt'
    
    # Test the script for errors
    Test=False

    # Set parameter based on test flag.
    if Test:
        samples=10
        notrial=10
        noofestimator=10
        Number_of_processor=1
    else:
        samples=600
        notrial=150
        noofestimator=1000
        Number_of_processor=1

    # Take user choice as input
    print("Please select one of the options:")
    print("1. ShapBasedOnETR+ShapBasedOnRFR")
    print("2. ShapBasedOnETR+ShapBasedOnRFR+SVR")
    usr_input = ''
    while usr_input not in ['1', '2']:
        usr_input = input("Enter Choice: ")

    # Ignore wanrings
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore")

    #Assign Seed
    np.random.seed(100)

    # Create output directory and file
    result_path="./"+FileName+"/"
    pathlib.Path(result_path).mkdir(parents=True, exist_ok=True)    
    Output_file=open(result_path+FileName+"_Results.txt","w")
    t0 = time.time()
    host_name = socket.gethostname() 
    print("Hostname :  ",host_name,file=Output_file)

    # Read from the file 
    with open(inputfile) as f:
        first_line = f.readline()
        second_line = f.readline()
        third_line = f.readline()
        fourth_line = f.readline()
   
    print("\nFile Location: \n",first_line)
    train_df = pd.read_csv(first_line.strip(),sep='\t')
    print("train_df Shape:" ,train_df.shape)
    train = train_df.values
    # print(pd.DataFrame(train).head(5))

    if fourth_line !="":
        # Read decoy list if exists
        decoy=pd.read_csv(fourth_line.strip(),sep='\t')
        decoy_list=decoy[decoy.Name.str.startswith('decoy')]["#ID"].tolist()
        train_df=train_df[train_df.columns[~train_df.columns.isin(decoy_list)]]
        # Read transcription factor list if exist
        tf=pd.read_csv(third_line.strip(),sep='\t',header=None)
        tf_list=list(np.setdiff1d( tf[0], decoy_list))

    # Create ML instances
    ML_instances = {}
    ML_instances["RFR"] = RandomForestRegressor(n_estimators=noofestimator,random_state=42, n_jobs=Number_of_processor)
    ML_instances["ETR"] =  ExtraTreesRegressor(n_estimators=noofestimator,random_state=42, n_jobs=Number_of_processor)    
    ML_instances["SVR"] =  LinearSVR(max_iter=noofestimator) 


    # %%

    # Extract Importance scores from RFR and ETR
    flag=0
    ML=["RFR","ETR"]
    for c in ML:    
        df_feature_importance_all=pd.DataFrame()
        print("ML Method Name: ",c)
        
        t = time.time()
        results = Parallel(n_jobs=64)(delayed(feature_importance2)(c,column) for column in notebook.tqdm(train_df.columns))
        df_feature_importance_all=pd.concat(results)
    
        df_feature_ranking=df_feature_importance_all.copy() 
        scaler = MinMaxScaler()
        scaler.fit(df_feature_ranking["feat_importance"].to_numpy().reshape(-1,1))
        z=scaler.transform(df_feature_ranking["feat_importance"].to_numpy().reshape(-1,1))
        df_feature_ranking["feat_Proba"]=z

        scaler.fit(df_feature_ranking["shap"].to_numpy().reshape(-1,1))
        z=scaler.transform(df_feature_ranking["shap"].to_numpy().reshape(-1,1))
        df_feature_ranking["shap_Proba"]=z
        df_feature_ranking.to_csv(result_path+c+".csv")

        print(c+" Time: ", time.time() - t,file=Output_file)      
        Output_file.flush()


    # %%
    # Extract Importance scores from SVR
    # print(usr_input)
    if int(usr_input)==2:
        flag2=0
        ML=[ "SVR"]
        for c in ML:    
            print("ML Method Name: ",c)
            
            t = time.time()
            results = Parallel(n_jobs=64)(delayed(feature_importance)(c,column) for column in notebook.tqdm(train_df.columns))    
        
            df_feature_importance_all_SVR=pd.concat(results) 
            scaler = MinMaxScaler()
            scaler.fit(df_feature_importance_all_SVR["feat_importance"].to_numpy().reshape(-1,1))
            z=scaler.transform(df_feature_importance_all_SVR["feat_importance"].to_numpy().reshape(-1,1))
            df_feature_importance_all_SVR["SVRProba"]=z
            df_feature_importance_all_SVR.to_csv(result_path+c+".csv")
            print("SVR Time: ", time.time() - t,file=Output_file)
            Output_file.flush()    


    # %%
    if int(usr_input)==1:
        print("1. ShapBasedOnETR+ShapBasedOnRFR Results:")
    else:
        print("2. ShapBasedOnETR+ShapBasedOnRFR+SVR Results:")
    evalresults(second_line,result_path,Output_file,FileName)
    print("Total Time: ", time.time() - t0,file=Output_file)
    Output_file.flush()
    Output_file.close()
    # %%



