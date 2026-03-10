import chainlit as cl
from src.model_query import query_rag


@cl.on_message
async def main(message: cl.Message):
    response = query_rag(message.content)
    await cl.Message(content=response).send()
