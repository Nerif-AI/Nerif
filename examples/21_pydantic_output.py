"""Example: Type-safe structured output with Pydantic models.

Demonstrates response_model parameter for automatic JSON Schema generation
and validated Pydantic object responses from LLMs.
"""

from typing import List, Optional
from pydantic import BaseModel
from nerif.model import SimpleChatModel
from nerif.utils import NerifFormat


# Define structured output types
class City(BaseModel):
    name: str
    country: str
    population: int
    famous_for: str


class TravelPlan(BaseModel):
    destination: City
    duration_days: int
    activities: List[str]
    budget_usd: Optional[int] = None


# Use response_model for automatic parsing
model = SimpleChatModel()

city = model.chat(
    "Tell me about Tokyo as a data object.",
    response_model=City,
)
print(f"City: {city.name}, {city.country}")
print(f"Population: {city.population:,}")
print(f"Famous for: {city.famous_for}")

# Nested models work too
plan = model.chat(
    "Create a 5-day travel plan for Paris.",
    response_model=TravelPlan,
)
print(f"\nTrip to {plan.destination.name} for {plan.duration_days} days")
print(f"Activities: {', '.join(plan.activities)}")

# You can also use FormatVerifierPydantic directly
from nerif.utils import FormatVerifierPydantic

verifier = FormatVerifierPydantic(City)
raw_json = '{"name": "London", "country": "UK", "population": 8900000, "famous_for": "Big Ben"}'
city2 = verifier(raw_json)
print(f"\nParsed: {city2.name} - {city2.famous_for}")

# NerifFormat.pydantic_parse handles markdown-wrapped JSON too
messy_output = (
    '```json\n{"name": "Berlin", "country": "Germany", "population": 3700000, "famous_for": "Brandenburg Gate"}\n```'
)
city3 = NerifFormat.pydantic_parse(messy_output, City)
print(f"Parsed from markdown: {city3.name}")
