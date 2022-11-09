import numpy as np
import pandas as pd
import boto3
import json
import os
import psycopg2
import ast
import csv
import io
from io import StringIO, BytesIO, TextIOWrapper
import gzip
from datetime import datetime, date
from s3_utils import *

bucket='fouryouandme-study-data'
key = 'bump/bodyport/wave_1/bodyport.csv.gz'

df = pandas_from_csv_s3(bucket, key=key, compression='gzip')
df.to_csv('results/bodyport.csv', index=False)