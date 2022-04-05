import json

with open('intents.json', 'r') as f:
    data = f.read()

data_dict = json.loads(data)
print(data_dict)

for ddict in data_dict['intents']:
    output_resp = ""
    ddict['responses'] = []
    ddict['responses'].append(output_resp)