import os
from pathlib import Path                 

from dotenv import load_dotenv          
from google import genai

from schema import RentalListing

load_dotenv()                            
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

instruction = Path("prompts/v1.txt").read_text(encoding="utf-8")
ad = Path("data/ads/001.txt").read_text(encoding="utf-8")
prompt = instruction + "\n\nAD:\n" + ad  

interaction = client.interactions.create(
    model="gemini-3.1-flash-lite",
    input=prompt,
    response_format={                     
        "type": "text",
        "mime_type": "application/json",
        "schema": RentalListing.model_json_schema(),  
    },
)

print(interaction.output_text)           


listing = RentalListing.model_validate_json(interaction.output_text)
print(listing)
print(interaction.usage)                 