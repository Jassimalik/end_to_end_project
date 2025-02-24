import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.abspath("src"))
# sys.append(os.path.abspath("src"))
from mlproject.exception import CustomException
from mlproject.logger import logging
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from mlproject.utils import save_object
from mlproject.components.data_ingestion import DataIngestion

@dataclass
class DataTransformationConfig:
    data_transform_config = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_tranformation_obj(self,X):
        try:
            num_features = X.select_dtypes(exclude="object").columns
            cat_features = X.select_dtypes(include="object").columns

            num_pipeline = Pipeline([
                ("imputer",SimpleImputer(strategy="mean")),
                ("Scaler",StandardScaler())
            ])

            cat_pipeline=Pipeline([
                ("impute",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoding",OneHotEncoder())
            ])

            logging.info("numerial and categorical column are transform")

            preprocessor = ColumnTransformer([
                ("numerical_pipeline",num_pipeline,num_features),
                ("cat_pipeline",cat_pipeline,cat_features)
            ])
            return preprocessor
        
        except Exception as ex:
            raise CustomException(ex,sys)
    

    
    def initate_data_transform(self,train_path,test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            logging.info('training and testing data are completed')
            target_column = "math_score"
            # X = train_data.drop(columns=[target_column])
            target_train_feature_data = train_data[target_column]
            input_train_feature_data = train_data.drop(columns=[target_column])
            preprocessing_obj = self.get_data_tranformation_obj(input_train_feature_data)
            
            target_test_feature_data = test_data[target_column]
            input_test_feature_data = test_data.drop(columns=[target_column])
            

            logging.info(
                    f"Applying preprocessing object on training dataframe and testing dataframe."
                )
            input_feature_train_arr = preprocessing_obj.fit_transform(input_train_feature_data)
            input_feature_test_arr = preprocessing_obj.fit_transform(input_test_feature_data)

            train_arr = np.c_[input_feature_train_arr,np.array(target_train_feature_data)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_test_feature_data)]

            logging.info('save train and test array')
            save_object(file_path=self.data_transformation_config.data_transform_config,obj=preprocessing_obj)

            return(
                train_arr,test_arr,self.data_transformation_config.data_transform_config
            )
        except Exception as ex:
            raise CustomException(ex,sys)
        
if __name__ =='__main__':
   obj = DataIngestion()
   train_path, test_path = obj.initiate_data_ingestion()
   trn = DataTransformation()
   tr_arr, test_arr, pre = trn.initate_data_transform(train_path,test_path)
   print(tr_arr)

    







