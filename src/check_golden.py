import json                          
from schema import RentalListing     

with open("data/golden.jsonl", encoding="utf-8") as f:
    for line in f:                 
        
        row = json.loads(line)
        
        
        RentalListing.model_validate(row["expected"])
        print(row["id"], "OK") 

print("Все строки golden set валидны")