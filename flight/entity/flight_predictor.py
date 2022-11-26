import os
import sys

from flight.exception import FlightException
from flight.util.util import load_object

import pandas as pd


class flightData:

    def __init__(self,
                Airline : str,
                Source : str,
                Destination : str,
                Total_Stops : int,
                journey_Date : int,
                journey_Month : int,
                Dep_hour : int,
                Dep_min : int,
                Arrival_hour : int,
                Arrival_min : int,
                Duration_hours : int,
                Duration_mins : int,
                Price : int = None
                 ):
        try:
            self.Airline = Airline
            self.Source = Source
            self.Destination = Destination
            self.Total_Stops = Total_Stops
            self.journey_Date = journey_Date
            self.journey_Month = journey_Month
            self.Dep_hour = Dep_hour
            self.Dep_min = Dep_min
            self.Arrival_hour = Arrival_hour
            self.Arrival_min = Arrival_min
            self.Duration_hours = Duration_hours
            self.Duration_mins = Duration_mins
            self.Price = Price
        except Exception as e:
            raise FlightException(e, sys) from e

    def get_flight_input_data_frame(self):

        try:
            flight_input_dict = self.get_flight_data_as_dict()
            return pd.DataFrame(flight_input_dict)
        except Exception as e:
            raise FlightException(e, sys) from e

    def get_flight_data_as_dict(self):
        try:
            input_data = {
                "Airline" : [self.Airline],
                "Source" : [self.Source],
                "Destination" : [self.Destination],
                "Total_Stops" : [self.Total_Stops],
                "journey_Date" : [self.journey_Date],
                "journey_Month" : [self.journey_Month],
                "Dep_hour" : [self.Dep_hour],
                "Dep_min" : [self.Dep_min],
                "Arrival_hour" : [self.Arrival_hour],
                "Arrival_min" : [self.Arrival_min],
                "Duration_hours" : [self.Duration_hours],
                "Duration_mins" : [self.Duration_mins],
                }
            return input_data
        except Exception as e:
            raise FlightException(e, sys)


class flightPredictor:

    def __init__(self, model_dir: str):
        try:
            self.model_dir = model_dir
        except Exception as e:
            raise FlightException(e, sys) from e

    def get_latest_model_path(self):
        try:
            folder_name = list(map(int, os.listdir(self.model_dir)))
            latest_model_dir = os.path.join(self.model_dir, f"{max(folder_name)}")
            file_name = os.listdir(latest_model_dir)[0]
            latest_model_path = os.path.join(latest_model_dir, file_name)
            return latest_model_path
        except Exception as e:
            raise FlightException(e, sys) from e

    def predict(self, X):
        try:
            model_path = self.get_latest_model_path()
            model = load_object(file_path=model_path)
            price = model.predict(X)
            return price
        except Exception as e:
            raise FlightException(e, sys) from e