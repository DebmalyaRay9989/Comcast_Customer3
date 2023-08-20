import requests

url = "https://text-analysis12.p.rapidapi.com/sentiment-analysis/api/v1.1"

payload = {
	"language": "english",
	"text": "Falcon 9’s first stage has landed on the Of Course I Still Love You droneship – the 9th landing of this booster"
}
headers = {
	"content-type": "application/json",
	"X-RapidAPI-Key": "ee0947a6afmshd9a0846869b0f80p12916fjsn610316852108",
	"X-RapidAPI-Host": "text-analysis12.p.rapidapi.com"
}

response = requests.request("POST", url, json=payload, headers=headers)

print(response.text)
print('***************************************************************************')
print('***************************************************************************')
print('***************************************************************************')
print('***************************************************************************')
print('***************************************************************************')
jsondata = response.json()
print(jsondata)
print('***************************************************************************')
print('***************************************************************************')
print('***************************************************************************')
print('***************************************************************************')
print('***************************************************************************')
print(jsondata.keys())

from pandas import json_normalize 
import requests
import json
import pandas as pd
textdata = json.loads(response.text)

res = json_normalize(textdata)

df = pd.DataFrame(res)
print(df.shape)
print(df.head(10))