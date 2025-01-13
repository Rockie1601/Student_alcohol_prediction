import os
import sys
from src.exception import CustomExceptions
from src.logger import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.utils import save_obj

@dataclass
class transformationconfig:
    preprocessor_file_path=os.path.join('dataholder','preprocessor.pkl')

class transformation:
    def __init__(self):
        self.transformation_config=transformationconfig()
    
    def data_transformation_initiation(self):
        try:
            numerical_features=['age','Medu','Fedu','traveltime','studytime','failures','famrel','freetime','goout','health','absences','G1','G2','G3']
            categorical_features=['school','sex','address','famsize','Pstatus','Mjob','Fjob','reason','guardian','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic']
            logging.info('finished categorising the data')

            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            logging.info("creating numerical pipeline")

            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("onehotencoding",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            logging.info('creating categorical pipeline')

            preprocessor=ColumnTransformer(
                [
                    ("numerical pipeline",num_pipeline,numerical_features),
                    ("categorical pipeline",cat_pipeline,categorical_features)
                ]
            )
            return preprocessor

        except Exception as e:
            raise CustomExceptions(e,sys)
    
    def inititate_data_tranform(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            preprocessor=self.data_transformation_initiation()
            logging.info('created a preprocessing object')

            target_column=['Dalc','Walc']

            input_train=train_df.drop(columns=target_column,axis=1)
            target_train=train_df[target_column]

            input_test=test_df.drop(columns=target_column,axis=1)
            target_test=test_df[target_column]

            logging.info('applying preprocessing')

            input_trans_train=preprocessor.fit_transform(input_train)
            input_trans_test=preprocessor.transform(input_test)

            train_data=np.c_[input_trans_train,np.array(target_train)]
            test_data=np.c_[input_trans_test,np.array(target_test)]
            logging.info('completed transformation')
            save_obj(
                file_path=self.transformation_config.preprocessor_file_path,
                obj=preprocessor
            )

            return train_data,test_data,self.transformation_config.preprocessor_file_path
        except Exception as e:
            raise CustomExceptions(e,sys)

