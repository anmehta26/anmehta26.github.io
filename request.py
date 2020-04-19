import requests

url = 'anmehta26.me'
r = requests.post(url,json={'Molecule SMILES':2})

print(r.json())
