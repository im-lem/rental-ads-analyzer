from enum import StrEnum
from pydantic import BaseModel, Field 

class PropertyType(StrEnum):         
    apartment = "apartment"
    house = "house"
    room = "room"
    unknown = "unknown"


class Currency(StrEnum):             
    UAH = "UAH"                       
    USD = "USD"
    EUR = "EUR"
    unknown = "unknown"


class Utilities(StrEnum):             
    included = "included"             
    separate = "separate"             
    partial = "partial"              
    unknown = "unknown"


class PetsPolicy(StrEnum):            
    yes = "yes"
    no = "no"
    negotiable = "negotiable"
    unknown = "unknown"


class LandlordType(StrEnum):          
    owner = "owner"
    agent = "agent"
    unknown = "unknown"


class Amenity(StrEnum):             
    furniture = "furniture"
    appliances = "appliances"
    air_conditioner = "air_conditioner"
    boiler = "boiler"
    washing_machine = "washing_machine"
    internet = "internet"
    parking = "parking"
    balcony = "balcony"


class TextLanguage(StrEnum):          
    uk = "uk"
    ru = "ru"
    en = "en"
    mixed = "mixed"


class Price(BaseModel):                       
    amount: float | None = Field(gt=0)       
    currency: Currency                        
    utilities: Utilities                      

class Location(BaseModel):                    
    city: str | None                         
    district: str | None                      


class RentalListing(BaseModel):               
    property_type: PropertyType
    rooms: int | None = Field(ge=1, le=10)    
    area_m2: float | None = Field(ge=5, le=500)
    price: Price                             
    location: Location                      
    pets_allowed: PetsPolicy
    landlord_type: LandlordType
    amenities: list[Amenity]                  
    text_language: TextLanguage


if __name__ == "__main__":
    from pydantic import ValidationError

    good = {
        "property_type": "apartment",
        "rooms": 2,
        "area_m2": 54.5,
        "price": {"amount": 15000, "currency": "UAH", "utilities": "separate"},
        "location": {"city": "Київ", "district": None},
        "pets_allowed": "no",
        "landlord_type": "owner",
        "amenities": ["furniture", "boiler"],
        "text_language": "uk",
    }
    print(RentalListing.model_validate(good))   # ← этой строки не хватало

    bad = {                                      # ← 4 пробела: внутри if
        "property_type": "flat",
        "rooms": 0,
        "area_m2": 54.5,
        "price": {"amount": -100, "currency": "грн", "utilities": "separate"},
        "location": {"city": "Київ"},
        "pets_allowed": "no",
        "landlord_type": "owner",
        "amenities": ["sauna"],
        "text_language": "uk",
    }
    try:                                         # ← 4 пробела: внутри if
        RentalListing.model_validate(bad)
    except ValidationError as e:
        print(e)