from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import src.embedding.embeddings as embeddings
from dotenv import load_dotenv
import os
import logging
import time

logger = logging.getLogger(__name__)

load_dotenv()

STYLE_PROMPTS = {
    "essay": "Apply academic essay writing standards: formal tone, clear thesis, structured argumentation.",
    "fantasy": "Apply creative fiction standards: vivid world-building, narrative voice, show-don't-tell.",
    "formal": "Apply formal writing standards: professional tone, precise language, no contractions.",
}

# Prompt optimierung
SUB_PROMPTS = {
    "grammar": "Focus only on grammar errors",
    "style": "Focus only on writing style",
    "clarity": "Focus only on clarity",
}

PROMPT_TEMPLATE = """
You are an expert writing coach specialising in {style_context}. Use the following writing guidelines 
to give specific, actionable feedback on the user's text.

Language: en-GB

Writing Guidelines (from style guides):
{context}

Your Task:
{focus_instruction}

User's Text:
{question}

Be specific: reference parts of the text directly.
"""

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma")


def get_db(chroma_path: str, collection_name: str):
    if not os.path.exists(chroma_path):
        raise FileNotFoundError(f"ChromaDB not found at '{chroma_path}'. ")

    embedding_function = embeddings.get_embedding_function(logger)
    db = Chroma(
        persist_directory=chroma_path,
        embedding_function=embedding_function,
        collection_name=collection_name,
    )

    count = db.get()  # if exists but empty
    if len(count["ids"]) == 0:
        raise ValueError(
            "ChromaDB is empty. " "Run generate_chroma.py first to populate it."
        )

    return db


def query_rag(user_text: str, style: str = "essay") -> str:

    logger.info("1. Received user query for RAG feedback.")

    if not user_text:
        return "Please provide some text to get feedback on."

    if len(user_text) > 50000:
        return (
            "Text is too long! It's over 5000 characters."
            "Please split it into smaller sections."
        )

    logger.info("2. Input validation passed. Proceeding with RAG process.")
    collection_name = style if style in STYLE_PROMPTS else "essay"
    style_context = STYLE_PROMPTS.get(collection_name, STYLE_PROMPTS["essay"])

    # issue: where chroma path
    try:
        db = get_db(CHROMA_PATH, collection_name)
    except (FileNotFoundError, ValueError) as e:
        return f"Database error: {e}"

    logger.info("3. RAG similarity search started.")
    # If needed. Maybe doing similarity seraches for a combination of chunks could do better.
    results = db.similarity_search_with_score(user_text, k=1)
    # More than 3 is extremely heavy for the mini model
    # Effectively no character limit with sentence transformer embedding model.
    # Nomic-embed-text has a limit of 1000 words ~5000 characters

    logger.info("4. Received similarity search results.")
    logger.info(f"++ Debugging similarity search | Collection '{collection_name}' ++")
    for doc, score in results:
        logger.info(f"Score: {score:.3f} | Source: {doc.metadata.get('source', '?')}")
        logger.debug(f"Content: {doc.page_content[:150]}")

    if not results:
        return "No relevant writing guidelines found."

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    # logger.info("5. Preparing prompt for LLM.")
    #
    # prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    # prompt = prompt_template.format(
    #     style_context=style_context, context=context_text, question=user_text
    # )

    logger.info("6. Invoking LLM with prepared prompt.")

    logger.info(f"Using model: {os.getenv('LOCAL_MODEL')}")

    # Well this doesn't work for Windows.
    model = OllamaLLM(
        model=os.getenv("LOCAL_MODEL", "qwen3.5:4b"),
        base_url="http://127.0.0.1:11434",
        temperature=0.3,
    )  # test different models qwen

    start = time.time()

    feedback_sections = {}
    for section, focus in SUB_PROMPTS.items():
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
            style_context=style_context,
            focus_instruction=focus,
            context=context_text,
            question=user_text,
        )
        logger.info(f"7. Running sub-prompt: {section}")

        try:
            feedback_sections[section] = model.invoke(prompt)
        except Exception as e:
            logger.error(f"LLM invocation failed: {e}")
            raise

    end = time.time()
    logger.info(f"8. LLM response time: {end - start:.2f} seconds")
    # logger.warning(f"LLM response: {format_feedback(feedback_sections)}")
    return format_feedback(feedback_sections)


def format_feedback(sections: dict) -> str:
    return (
        f"### Grammar\n{sections['grammar']}\n\n"
        f"### Style\n{sections['style']}\n\n"
        f"### Clarity\n{sections['clarity']}"
    )
