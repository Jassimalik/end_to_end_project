import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath("src"))
from mlproject.utils import load_data
from mlproject.exception import CustomException
from mlproject.logger import logging

class DataPredition:
    def __init__(self):
        pass
    def predition(self,feature):
        try:
            model_path = os.path.join('artifacts','model.pkl')
            preprosser_path = os.path.join('artifacts','preprocessor.pkl')
            model = load_data(model_path)
            print("jassi")
            print(model)
            preprosser = load_data(preprosser_path)
            transform_data = preprosser.transform(feature)
            predict = model.predict(transform_data)
            logging.info("Predict the output for new values")
            return predict
        except Exception as ex:
            raise CustomException(ex,sys)
        
class CustomData:
    def __init__(self,
                 gender :str,
                 race_ethnicity :str,
                 parental_level_of_education :str,
                 lunch :str,
                 test_preparation_course :str,
                 reading_score :int,
                 writing_score :int
                 ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def data_as_dataframe(self):
        try:
            data = {
                'gender' : [self.gender],
                'race_ethnicity' : [self.race_ethnicity],
                'parental_level_of_education' : [self.parental_level_of_education],
                'lunch' : [self.lunch],
                'test_preparation_course' : [self.test_preparation_course],
                'reading_score' :[self.reading_score],
                'writing_score' : [self.writing_score]
            }
            return pd.DataFrame(data)
        except Exception as ex:
            raise CustomException(ex,sys)
if __name__ =='__main__': 
    custom_data = CustomData(
        gender='male',
        race_ethnicity='group B',
        parental_level_of_education="bachelor's degree",
        lunch='standard',
        test_preparation_course='completed',
        reading_score=50,
        writing_score=60
    )
    input_df = custom_data.data_as_dataframe()
    prediction = DataPredition().predition(input_df)
    print(prediction)