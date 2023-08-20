import requests

url = "https://bloomberg-market-and-financial-news.p.rapidapi.com/market/auto-complete"

##querystring = {"query":"infosys"}
querystring = {"query":"company"}

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

print(type(jsondata['quote']))
print(type(jsondata['news']))
import pandas as pd
df = pd.DataFrame(jsondata['quote'])
df2 = pd.DataFrame(jsondata['news'])
print('***************************************************************************')
print(df.info())
print(df2.info())
print('***************************************************************************')
print(df.head(3))
print(df2.head(3))
print('***************************************************************************')
print(df.shape)
print(df2.shape)
print('***************************************************************************')
print(df.columns)
print(df2.columns)
print('***************************************************************************')
df3 = pd.concat([df, df2], axis = 1)
print(df3.columns)
print(df3.shape)
print('***************************************************************************')
