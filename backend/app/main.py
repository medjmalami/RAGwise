from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000"
    ],  # your Next.js dev origin — adjust port if different
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
