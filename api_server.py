from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict
from query_backend import process_user_query

app = FastAPI()

# Define request body model
class QueryRequest(BaseModel):
    query: str

@app.post("/process_query")
async def handle_query(request: QueryRequest):
    # Call your real query processing function with the input query string
    result: Dict[str, Any] = process_user_query(request.query)

    # Return the JSON-serializable dictionary as response
    return result
