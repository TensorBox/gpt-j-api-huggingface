import requests, json
from requests.api import head

payload = { 
    "prompt" : "In a shocking finding, scientist discovered", 
    "max_length" : 100
}
headers = {'Content-type': 'application/json'}
response = requests.post("http://0.0.0.0:5000/generate", data=json.dumps(payload), headers=headers).json()
print(response)