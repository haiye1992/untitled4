import urllib3
import json
import scrapy

http=urllib3.PoolManager()
r=http.request('GET','http://www.baidu.com')
print(r.data)
json.loads(r.data.decode('utf-8'))