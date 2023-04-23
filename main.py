"""Main entrypoint for the app."""
import logging
import os
import pickle
import pinecone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from langchain.vectorstores import VectorStore
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings

from callback import QuestionGenCallbackHandler, StreamingLLMCallbackHandler
from query_data import get_chain
from schemas import ChatResponse

import logging
log = logging.getLogger("talk-to-rudin")
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - > %(message)s')

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY is not None, "OPENAI_API_KEY is not set"

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
assert PINECONE_API_KEY is not None, "PINECONE_API_KEY is not set"

PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
assert PINECONE_ENVIRONMENT is not None, "PINECONE_ENVIRONMENT is not set"

PINECONE_INDEX = os.getenv("PINECONE_INDEX")
assert PINECONE_INDEX is not None, "PINECONE_INDEX is not set"


app = FastAPI()
templates = Jinja2Templates(directory="templates")
vectorstore: Optional[VectorStore] = None


@app.on_event("startup")
async def startup_event():
    global vectorstore
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

    if PINECONE_INDEX not in pinecone.list_indexes():
        log.error(f"Index {PINECONE_INDEX} does not exist. Please create it first.")
    else:
        try:
            log.info(f"Connecting to existing index {PINECONE_INDEX}")
            vectorstore = Pinecone(
                index=pinecone.Index(PINECONE_INDEX),
                embedding_function=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY).embed_query,
                text_key="text")
            log.info(f"Connected to index {PINECONE_INDEX} successfully")
        except Exception as e:
            log.error(f"Error connecting to index {PINECONE_INDEX}: {e}")


@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    question_handler = QuestionGenCallbackHandler(websocket)
    stream_handler = StreamingLLMCallbackHandler(websocket)
    chat_history = []
    qa_chain = get_chain(vectorstore, question_handler, stream_handler)
    # Use the below line instead of the above line to enable tracing
    # Ensure `langchain-server` is running
    # qa_chain = get_chain(vectorstore, question_handler, stream_handler, tracing=True)

    while True:
        try:
            # Receive and send back the client message
            question = await websocket.receive_text()
            resp = ChatResponse(sender="you", message=question, type="stream")
            await websocket.send_json(resp.dict())

            # Construct a response
            start_resp = ChatResponse(sender="bot", message="", type="start")
            await websocket.send_json(start_resp.dict())

            result = await qa_chain.acall(
                {"question": question, "chat_history": chat_history}
            )
            log.info(result)
            chat_history.append((question, result["answer"]))

            end_resp = ChatResponse(sender="bot", message="", type="end")
            await websocket.send_json(end_resp.dict())
        except WebSocketDisconnect:
            log.info("websocket disconnect")
            break
        except Exception as e:
            log.error(e)
            resp = ChatResponse(
                sender="bot",
                message="Sorry, something went wrong. Try again.",
                type="error",
            )
            await websocket.send_json(resp.dict())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
