import sys
import os
sys.path.append(os.path.abspath("src"))  # Ensure Python finds 'src'
from mlproject.logger import logging
from mlproject.exception import CustomException
from mlproject.components.data_ingestion import DataIngestion
from mlproject.components.data_ingestion import DataIngestionConfig

if __name__=="__main__":
    logging.info("The execution start")

    try:
        data_ingestion=DataIngestion()
        data_ingestion.initiate_data_ingestion()
    except Exception as ex:
        raise CustomException(ex,sys)
