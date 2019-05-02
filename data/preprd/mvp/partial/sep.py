import json
import numpy as np


for r in [10,20,30,40,50,60,70,80,90]:
    data = json.load(open('partial'+str(r)))
    json.dump({'history':data['history'], 'label':data['label']}, open('partial'+str(r)))

