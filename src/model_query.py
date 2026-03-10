from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from src.embeddings import get_embedding_function_local
from dotenv import load_dotenv
import os
import logging

logger = logging.getLogger(__name__)

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma")

STYLE_INSTRUCTIONS = {
    "essay": "Apply academic essay writing standards: formal tone, clear thesis, structured argumentation.",
    "fantasy": "Apply creative fiction standards: vivid world-building, narrative voice, show-don't-tell.",
    "formal": "Apply formal writing standards: professional tone, precise language, no contractions.",
}

PROMPT_TEMPLATE = """
You are an expert writing coach. Use the following writing guidelines 
to give specific, actionable feedback on the user's text.

Writing Guidelines (from style guides):
{context}

User's Text:
{question}

Provide feedback on: grammar, style, tone, and clarity.
Be specific — reference parts of the text directly.
"""
# verbessere den style - wenn user erwähnt essay und style dann system prompt updaten
# select : essay, fantasy, formal
load_dotenv()


def get_db():
    if not os.path.exists(CHROMA_PATH):
        raise FileNotFoundError(
            f"ChromaDB not found at '{CHROMA_PATH}'. "
            "Ask your friend to share the chroma/ folder."
        )

    embedding_function = get_embedding_function_local()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    count = db.get()  # if exists but empty
    if len(count["ids"]) == 0:
        raise ValueError(
            "ChromaDB is empty. " "Run generate_chroma.py first to populate it."
        )

    return db


def query_rag(user_text: str) -> str:

    if not user_text:
        return "Please provide some text to get feedback on."

    if len(user_text) > 5500:
        return (
            "Text is too long! It's over 5000 characters."
            "Please split it into smaller sections."
        )

    try:
        db = get_db()
    except (FileNotFoundError, ValueError) as e:
        return f"Database error: {e}"

    results = db.similarity_search_with_score(user_text, k=5)

    logger.info("++ Debugging similarity_search_with_score ++")
    for doc, score in results:
        logger.info(f"Score: {score:.3f} | Source: {doc.metadata.get('source', '?')}")
        logger.debug(f"Content: {doc.page_content[:150]}")

    if not results:
        return "No relevant writing guidelines found."

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=user_text)

    model = OllamaLLM(
        model=os.getenv("LOCAL_MODEL", "llama3:8b")
    )  # teest different models qwen
    response = model.invoke(prompt)
    return response

if __name__ == "__main__":
    #
    # print("Debugging: ChromaDB Connection")
    # db = Chroma(
    #     persist_directory=CHROMA_PATH, embedding_function=get_embedding_function_local()
    # )
    # count = db.get()
    # print(f"Connected! Documents in DB: {len(count['ids'])}")
    #
    # print("\nDebugging: Retrieval")
    # TEST_TEXT = """I am writing this comment to complain about the worrying increase in car and lorry traffic in our
    # town lately.
    # As far as I am concerned, the main reason for this issue is that a 40-block building is being
    # constructed in our neighborhood. Lorries need to function at full speed to carry necessary material
    # for the construction. Moreover, the main road that leads to the center of the city is banned for
    # decoration of the coming festival. People whose workplaces are located in the center have no choice
    # but to drive through our town to their work everyday. Therefore, our town, which used to be quiet
    # and peaceful, has become noisier. The citizens in the town not only endure the syren that is made by
    # the vehicles, but also suffer from illnesses caused by air-polluted, especially the elderly. They cannot
    # sleep well at night since the lorries works 24 hours per day. Everything is dusty because of carelessly
    # covered containers of the lorries. We cannot let our children play at the playground of the town for
    # the fear that they can put themselves in danger. Futhermore, we are usually late for work and school
    # because of traffic jams in crush hours.
    # As a citizen of the town, I think the authority should take action to reduce the amount of traffic that
    # travels to our town every day. Lorries must be banned in crush hours. Moreover, the container of lorries should be covered carefully in order not to drop any material that they are carrying. Last but
    # not least, people should be shown other ways that lead to the center so that they do not put pressure
    # on our frastructure.
    # I hope that my complain can be taken into consideration so that our town can become an enjoyable
    # place to live again."""
    #
    # print("\nDebugging: Retrieval")
    # results = db.similarity_search_with_score(TEST_TEXT, k=3)
    # for doc, score in results:
    #     print(f"Score: {score:.3f}")
    #     print(f"Source: {doc.metadata.get('source', '?')}")
    #     print(f"Content: {doc.page_content[:150]}...")
    #     print()
    #
    # print("\nDebugging: Prompt Construction")
    # response = query_rag(TEST_TEXT)
    # print(response)
