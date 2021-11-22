import requests

payload = { 
    "prompt" : "In a shocking finding, scientist discovered", 
    "max_length" : 100
}
response = requests.post("http://0.0.0.0:5000/generate", params=payload).json()
print(response)