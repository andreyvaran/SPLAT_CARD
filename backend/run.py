import uvicorn
from config.app import app_config

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=app_config.HOST,
        port=app_config.PORT,
        workers=app_config.WORKERS,
    )
