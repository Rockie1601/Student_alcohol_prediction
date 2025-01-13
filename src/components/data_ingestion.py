import os
import sys
from src.exception import CustomExceptions
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import pandas as pd
from src.components.data_transformation import transformationconfig
from src.components.data_transformation import transformation
from src.components.model_trainer import modeltrainer


logging.basicConfig(level=logging.INFO)

@dataclass
class IngestionConfig:
    train_path: str = os.path.join('dataholder', 'train.csv')
    test_path: str = os.path.join('dataholder', 'test.csv')
    raw_path: str = os.path.join('dataholder', 'raw.csv')

class Ingestion:
    def __init__(self):
        self.ingestion_config = IngestionConfig()
        logging.info('IngestionConfig initialized.')

    def data_ingestion(self):
        logging.info('Started the data ingestion process.')
        try:
            df = pd.read_csv('/Users/vijayrakeshreddybandela/Documents/machine_learning(vs)/student_alcohol_prediction/src/data/dataset.csv')
            logging.info('Read data from dataset.csv.')
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_path), exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_path, index=False, header=True)
            logging.info('Created raw CSV file at dataholder.')

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info('Split the dataset into train and test sets.')

            train_set.to_csv(self.ingestion_config.train_path, index=False, header=True)
            logging.info('Created train CSV file at dataholder.')

            test_set.to_csv(self.ingestion_config.test_path, index=False, header=True)
            logging.info('Created test CSV file at dataholder.')

            return self.ingestion_config.raw_path, self.ingestion_config.train_path, self.ingestion_config.test_path
        
        except Exception as e:
            logging.error('Error during data ingestion process.', exc_info=True)
            raise CustomExceptions(e, sys)


if __name__ == "__main__":
    ingestion = Ingestion()
    train_path,test_path,_=ingestion.data_ingestion()

    trans=transformation()
    train_arr,test_arr,_=trans.inititate_data_tranform(train_path,test_path)

    model_trainer=modeltrainer()
    model_trainer.initiate_model_training(train_arr,test_arr)