
# we  are current;y setting up the basic data like importing libraries
import pandas as pd 
import numpy as np 

# with help of os library we are enabling for storage and usage of data
import os
for dirname, _, filenames in os.walk(r'D:\retail-sales-analytics\salesAnalytics'):
    for filename in filenames:
        print(":::::",os.path.join(dirname, filename))