from flask import Flask
from flask import jsonify
import numpy as np
import pandas as pd
import random
import time
import joblib

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello_world():

    scaler=joblib.load('std_scaler.bin')
    lm = joblib.load('GaussianNBjb.sav')
    json_file = {}
    dicti={
        0:'A-B',
        1:'A-B-C',
        2:'A-B-C-G',
        3:'A-B-G',
        4:'A-G',
        5:'B-C',
        6:'B-C-G',
        7:'B-G',
        8:'C-A',
        9:'C-A-G',
        10:'C-G',
        11:'Normal',
        }
    while(True):
        Va=random.uniform(9.568218, 6379.421118)
        Vb=random.uniform(9.709918, 6379.496404)
        Vc=random.uniform(9.674157, 6379.383102)
        
        Ia=random.uniform(120, 7536.392613)
        Ib=random.uniform(120, 7768.591696)
        Ic=random.uniform(120, 7609.606572)
        
        randarray=[Va,Vb,Vc,Ia,Ib,Ic]
        testarr=scaler.transform([randarray])
        predic=lm.predict(testarr)
        # print(dicti[predic[0]])
        json_file['query'] = dicti[predic[0]]
        time.sleep(3)
        return jsonify(json_file)


if __name__ == '__main__':
    app.run()