import json
import pickle
import numpy as np

__locations = None
__data_cols = None
__model = None




def get_estimated_price(location,sqft,bhk,bath):
    try:
        loc_index = __data_cols.index(location.lower())
    except:
        loc_index = -1
    x = np.zeros(len(__data_cols))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    x[loc_index] = 1

    return round(__model.predict([x])[0],2)

def get_location_names():
    return __locations



def load_artifacts():
    print("Loading saved artifacts")
    
    
    
    #columns data
    with open("../model/columns.json","r") as f:
        global __data_cols
        global __locations
        global __model
        __data_cols = json.load(f)['data_columns']
        __locations = __data_cols[3:]


    #model
    with open("../model/bangalore_home_price_model.pickle",'rb') as f:
        __model = pickle.load(f)
    
    print("Loaded cols and model into the memory")


if __name__ =="__main__":
    print(get_location_names())
    print(get_estimated_price('1st Phase JP Nagar',5000,3,3.0))


    