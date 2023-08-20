import requests

url = "https://bloomberg-market-and-financial-news.p.rapidapi.com/news/list-by-region"

querystring = {"id":"europe-home-v3"}

headers = {
	"X-RapidAPI-Key": "ee0947a6afmshd9a0846869b0f80p12916fjsn610316852108",
	"X-RapidAPI-Host": "bloomberg-market-and-financial-news.p.rapidapi.com"
}

response = requests.request("GET", url, headers=headers, params=querystring)

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