import chainlit as cl
from src.model_query import query_rag
import logging

logging.basicConfig(level=logging.DEBUG)


@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name="Essay", markdown_description="Academic essay writing feedback"
        ),
        cl.ChatProfile(
            name="Fantasy", markdown_description="Creative fiction writing feedback"
        ),
        cl.ChatProfile(
            name="Formal", markdown_description="Professional/formal writing feedback"
        ),
    ]


@cl.on_chat_start
async def on_start():
    profile = cl.user_session.get("chat_profile")  # "Essay", "Fantasy", or "Formal"
    style = profile.lower() if profile else "essay"
    cl.user_session.set("style", style)
    await cl.Message(
        content=f"Senging your text for **{profile}** writing feedback."
    ).send()


@cl.on_message
async def main(message: cl.Message):
    style = cl.user_session.get("style", "essay")
    response = query_rag(message.content, style=style)
    await cl.Message(content=response).send()
