# Put the code for your API here.
import uvicorn
from src.api import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

