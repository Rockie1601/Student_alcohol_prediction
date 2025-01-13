import pandas as pd
import os
import sys
from src.exception import CustomExceptions

try:
    df_math=pd.read_csv('/Users/vijayrakeshreddybandela/Documents/machine_learning(vs)/student_alcohol_prediction/src/data/student-mat.csv')
    df_por=pd.read_csv('/Users/vijayrakeshreddybandela/Documents/machine_learning(vs)/student_alcohol_prediction/src/data/student-por.csv')

    data_path=os.path.join('src/data','dataset.csv')

    df=pd.merge(df_math, df_por, how = 'outer')
    df.to_csv(data_path,index=False,header=True)
except Exception as e:
    raise CustomExceptions(e,sys)
