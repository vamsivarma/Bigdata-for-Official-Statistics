import json
import os
import requests

from flask import Flask
from flask import request
from flask import make_response

# Flask app should start in global layout
app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def webhook():
    req = request.get_json(silent=True, force=True)
    print(json.dumps(req, indent=4))
    
    res = makeResponse(req)
    
    res = json.dumps(res, indent=4)
    # print(res)
    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'
    return r

def makeResponse(req):
    result = req.get("queryResult")
    parameters = result.get("parameters")
    city = parameters.get("geo-city")
    date = parameters.get("date")
    print("City: " + city)
    date = str(date.split("T")[0])
    print("Date: " + date)

    r=requests.get('http://api.openweathermap.org/data/2.5/forecast?q='+city+'&appid=b3c31d1f1068b0c789227c8bdb5c9ba0')
    json_object = r.json()
    weather=json_object['list']
    condition = ""
    for i in range(0,30):
        if date in weather[i]['dt_txt']:
            condition= weather[i]['weather'][0]['description']
            break
    speech = "The forecast for " + city + " for " + date + " is " + condition
    return {
        "fulfillmentText": speech,
        "fulfillmentMessages": [],
        "source": "dl-dialogflow-webhook"
    }

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    print("Starting app on port %d" % port)
    app.run(debug=False, port=port, host='0.0.0.0')

















