import json
import os
import requests

from flask import Flask
from flask import request
from flask import make_response

import conf

from keras.models import Sequential, load_model
from keras.layers import Dense

from train import confs, get_params
import numpy as np

import os

# Metadata for different companies
# Creating models based on company list
comp_list = ['Alibaba', 'Amazon', 'Apple', 'Dell', 'Facebook', 'Google', 'Microsoft', 'Tesla', 'Twitter', 'Wallmart']

optimal_epochs_map = {
    'Alibaba': 45,
    'Amazon': 450,
    'Apple': 50,
    'Dell': 20,
    'Facebook': 55,
    'Google': 410,
    'Microsoft': 45,
    'Tesla': 150,
    'Twitter': 20,
    'Wallmart': 40
}

predict_map = {
    'Alibaba': [[168.55,171.95,168.00,171.83,12801800], 171.52],
    'Amazon': [[1643.34,1665.26,1642.50,1658.81,4453100], 1640.26],
    'Apple': [[172.86,175.08,172.35,174.18,36101600], 174.24],
    'Dell': [[51.27,51.60,50.64,51.07,2754700], 50.94],
    'Facebook': [[169.15,171.98,168.69,171.16,22557000], 170.49],
    'Google': [[1124.84,1146.85,1117.25,1145.99,3531900], 1115.23],
    'Microsoft': [[106.06,107.27,105.96,107.22,27325400], 106.03],
    'Tesla': [[312.98,315.30,301.88,312.89,7352100], 321.35],
    'Twitter': [[34.29,34.57,33.92,34.37,17610200], 34.16],
    'Wallmart': [[95.25,95.94,95.02,95.60,6099900], 95.64]
}

# Flask app should start in global layout
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    
    req = request.get_json(silent=True, force=True)
    #print(json.dumps(req, indent=4))
    res = makeResponse(req)
    
    res = json.dumps(res, indent=4)
    # print(res)
    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'
    return r

def makeResponse(req):
    result = req.get("queryResult")
    parameters = result.get("parameters")
    cname =  parameters.get("company") #"Apple"
    date =  parameters.get("date") #"07-02-2019"
    #print("City: " + city)
    date = str(date.split("T")[0])
    #print("Date: " + date)

    pred_result = do_prediction(cname, optimal_epochs_map[cname], predict_map[cname])
    
    prediction_string = "The predicted stock price of " + cname + " on " + date + " is "
    prediction_string += str(pred_result['predicted']) + " and actual value is "
    prediction_string += str(pred_result['actual']) + " with error and error rate as "
    prediction_string += str(pred_result['error']) + " and " + str(pred_result['errorFraction'])

    return {
        "fulfillmentText": prediction_string,
        "fulfillmentMessages": [],
        "source": "dl-prediction-webhook"
    }

# Alibaba
# 2019-02-05,168.550003,171.949997,168.000000,171.830002,171.830002,12801800
# 2019-02-06,171.860001,173.089996,169.990005,171.520004,171.520004,11263300
# [[168.55,171.95,168.00,171.83,12801800], 171.52]

# Amazon
# 2019-02-05,1643.339966,1665.260010,1642.500000,1658.810059,1658.810059,4453100
# 2019-02-06,1670.750000,1672.260010,1633.339966,1640.260010,1640.260010,3935000
# [[1643.34,1665.26,1642.50,1658.81,4453100], 1640.26]

# Apple
# 2019-02-05,172.860001,175.080002,172.350006,174.179993,174.179993,36101600
# 2019-02-06,174.649994,175.570007,172.850006,174.240005,174.240005,28220700
# [[172.86,175.08,172.35,174.18,36101600], 174.24]

# Dell
# 2019-02-05,51.270000,51.599998,50.639999,51.070000,51.070000,2754700
# 2019-02-06,51.150002,51.910000,50.840000,50.939999,50.939999,2836200
# [[51.27,51.60,50.64,51.07,2754700], 50.94]

# Facebook
# 2019-02-05,169.149994,171.979996,168.690002,171.160004,171.160004,22557000
# 2019-02-06,171.199997,172.470001,169.270004,170.490005,170.490005,13264400
# [[169.15,171.98,168.69,171.16,22557000], 170.49]

# Google
#2019-02-05,1124.839966,1146.849976,1117.248047,1145.989990,1145.989990,3531900
# 2019-02-06,1139.569946,1146.989990,1112.810059,1115.229980,1115.229980,2026849
# [[1124.84,1146.85,1117.25,1145.99,3531900], 1115.23]

# Microsoft
# 2019-02-05,106.059998,107.269997,105.959999,107.220001,107.220001,27325400
# 2019-02-06,107.000000,107.000000,105.529999,106.029999,106.029999,20598500
# [[106.06,107.27,105.96,107.22,27325400], 106.03]

# Tesla
# 2018-09-04,296.940002,298.190002,288.000000,288.950012,288.950012,8350500
# [[296.94,298.19,288.00,288.95,8350500], 280.74]
# 2019-02-04,312.980011,315.299988,301.880005,312.890015,312.890015,7352100
# 2019-02-05,312.489990,322.440002,312.250000,321.350006,321.350006,6737100
# [[312.98,315.30,301.88,312.89,7352100], 321.35]

# Twitter
# 2019-02-05,34.290001,34.570000,33.919998,34.369999,34.369999,17610200
# 2019-02-06,35.049999,35.250000,33.750000,34.160000,34.160000,33687800
# [[34.29,34.57,33.92,34.37,34.37,17610200], 34.16]

# Wallmart
# 2019-02-05,95.250000,95.940002,95.019997,95.599998,95.599998,6099900
# 2019-02-06,95.430000,96.010002,95.220001,95.639999,95.639999,4263700
# [[95.25,95.94,95.02,95.60,6099900], 95.64]

def do_prediction(cname, cepochs, cpred):

    # this is the same as input_shape to our LSTM models
    # (num of past days of data to use, num of metrics to use)
    data_shape = (1,5)

    # Here we have input data for 04.09.2018
    # that we will base our prediction on,
    # and closing price on the next day (on 5th)
    # for validation.
    # [Open,High,Low,Close(t),Volume], Close (t+1)
    to_predict = cpred

    # Get comman line params.
    # name, epochs, batches, _=get_params(script='predict.py')
    
    name = "default"
    epochs = cepochs
    batches = 1
    
    model = confs[name]
    
    mname = 'models/'+ cname + '/model-%s-%d-%d.h5' % (name, epochs, batches)
    # Loading the model.
    if os.path.exists(mname):
        model=load_model(mname)
        print('Model loaded!')
    else:
        print("Can't find %s model, train it first using 'train.py %s %d %d'" % (mname, name, epochs, batches))
    p=np.array(to_predict[0])
    # Convert data into the "right format".
    p=np.reshape(p, (batches, data_shape[0], data_shape[1]))
    # Get the expected price for validation.
    c=to_predict[0][1]
    # Again here we need to specify the batch_size.
    x=model.predict(p, batch_size=batches)
    # We have just one prediction.
    x=x[0][0]

    #print('Prediction for ' + cname)
    #print('Predicted $%.2f, actual $%.2f, error $%.2f (%.2f%%)' % (x, c, x-c, abs((x-c)*100/c)))
    results = {}
    results['predicted'] = float("{0:.2f}".format(x))
    results['actual'] = float("{0:.2f}".format(c))
    results['error'] = float("{0:.2f}".format(x-c))
    results['errorFraction'] = float("{0:.2f}".format(abs((x-c)*100/c)))
    
    return results

'''
for cname in comp_list:
    do_prediction(cname, optimal_epochs_map[cname], predict_map[cname])
'''

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    print("Starting app on port %d" % port)
    app.run(debug=False, port=port, host='0.0.0.0')