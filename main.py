# Put the code for your API here.
import uvicorn
import os
from src.api import app

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  
    uvicorn.run(app, host="0.0.0.0", port=port)


