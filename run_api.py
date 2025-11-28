# run_api.py
# start the aksum credit risk api

import uvicorn
import sys
import os

sys.path.append("api")

from api.main import app
import config

print("")
print("=" * 50)
print("AKSUM CREDIT RISK ENGINE")
print("Starting API Server")
print("=" * 50)
print("")
print("Company: " + config.COMPANY_NAME)
print("Website: aksum.co.in")
print("")
print("API will be available at:")
print("  http://" + config.API_HOST + ":" + str(config.API_PORT))
print("")
print("API Documentation:")
print("  http://" + config.API_HOST + ":" + str(config.API_PORT) + "/docs")
print("")
print("Press CTRL+C to stop the server")
print("")
print("=" * 50)
print("")

if __name__ == "__main__":
    
    # run the api
    uvicorn.run(
        "api.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True
    )