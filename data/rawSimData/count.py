import json

with open("456Cycles20Dist.json", "r") as f:
    data = json.load(f)
    
d = False
k = data["features"][0][0]
for i in data["features"]:
    if i[0] != k:
        print(True)
    
print(False)
