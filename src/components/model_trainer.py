import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from  sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from src.logger import logging
from src.exception import CustomExceptions
from src.utils import save_obj
from src.utils import evaluate_model
from sklearn.metrics import r2_score

@dataclass
class modeltrainerconfig:
    trained_model_path:str=os.path.join('dataholder','model.pkl')

class modeltrainer:
    def __init__(self):
        self.trained_model_path=modeltrainerconfig()

    def initiate_model_training(self,train_arr,test_arr):
        try:
            logging.info('loading all the training algorithms')
            X_train,y_train,X_test,y_test=train_arr[:,:-1],train_arr[:,-1],test_arr[:,:-1],test_arr[:,-1]

            models={
                "CatBoostRegressor":CatBoostRegressor(),
                "AdaBoostRegressor":AdaBoostRegressor(),
                "GradientBoostingRegressor":GradientBoostingRegressor(),
                "RandomForestRegressor":RandomForestRegressor(),
                "LinearRegression":LinearRegression(),
                "DecisionTreeRegressor":DecisionTreeRegressor(),
                "KNeighborsRegressor":KNeighborsRegressor(),
                "XGBRegressor":XGBRegressor()
            }
            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model=models[best_model_name]
            if best_model_score < 0.6:
                raise CustomExceptions("No best model")
            
            logging.info("Completed the training process")
            logging.info("found the best model {}".format(best_model))

            save_obj(
                file_path=self.trained_model_path.trained_model_path,
                obj=best_model
            )
            predicted=best_model.predict(X_test)
            r_score=r2_score(y_test,predicted)
            accu=r_score*100
            logging.info("best models accuracy score is {}".format(accu))
            print(accu)
            return r_score
        


        except Exception as e:
            raise CustomExceptions(e,sys)
        