## this file is particular for model training

import os 
import sys
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor       
)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree  import DecisionTreeRegressor
from sklearn.metrics import r2_score


from xgboost import XGBRFRegressor

from src.utlis import save_object
from src.utlis import evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()



    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting Dependent and Independent variables from train and test data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            ## defieing the list of model
            models={
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "XGBRegressor":XGBRFRegressor(),
                "CatBoosting Regressor":CatBoostRegressor(verbose=False),
                "AdaBoost Regressor":AdaBoostRegressor()
            }


            params={
                "Random Forest":{
                    "n_estimators":[8,16,32,64,128,256]
                    # "max_depth":
                },

                "Decision Tree":{
                    "criterion":["squared_error","friedman_mse"],
                    # "max_depth":[8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    "learning_rate":[0.1,0.01,0.05],    
                    "subsample":[0.6,0.7,0.9],
                    "n_estimators":[8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "K-Neighbors Regressor":{
                    "n_neighbors":[5,7,9,11]    
                },
                "XGBRegressor":{
                    "learning_rate":[0.1,0.01,0.05],    
                    "n_estimators":[8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    "depth":[6,8,10],
                    "learning_rate":[0.1,0.01,0.05],    
                    "iterations":[30,50,100]
                },
                "AdaBoost Regressor":{
                    "learning_rate":[0.1,0.01,0.05],    
                    "n_estimators":[8,16,32,64,128,256]
                }


            }

            model_report:dict=evaluate_models(X_train=X_train
                                            ,y_train=y_train
                                            ,models=models
                                            ,X_test=X_test
                                            ,y_test=y_test
                                            ,params=params
                                            )
            
            best_model_score= max(sorted(model_report.values()))

            best_model_name= list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model= models[best_model_name]



            if best_model_score<0.6:
                raise Exception("No Best Model Found")
            
            logging.info(f"Best found model on both training and testing dataset {best_model_name} with r2 score of {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square=r2_score(y_test,predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e,sys)
