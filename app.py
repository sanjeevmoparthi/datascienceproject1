from src.datascienceproject1.logger import logging
from src.datascienceproject1.exception import CustomException
from src.datascienceproject1.components.data_ingestion import DataIngestion
from src.datascienceproject1.components.data_ingestion import DataIngestionConfig
from src.datascienceproject1.components.data_transformation import DataTransformationConfig,DataTransformation
from src.datascienceproject1.components.model_tranier import MOdelTrainer,MOdelTrainerConfig
import sys


if __name__=="__main__":
    logging.info("The execution has started")

    try:
        #data ingestion
        data_ingestion=DataIngestion()
        train_data_path,test_data_path =  data_ingestion.initiate_data_ingestion()

        # data transformation
        data_transformation_config = DataTransformationConfig()
        data_transformation = DataTransformation()
        train_arr,test_arr,_= data_transformation.initiate_data_transormation(train_data_path,test_data_path)


        ### MOdel training:
        model_trainer = MOdelTrainer()
        print( model_trainer.initiate_model_trainer(train_arr,test_arr))

    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)
