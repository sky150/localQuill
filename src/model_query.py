from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import src.embedding.embeddings as embeddings
from dotenv import load_dotenv
import os
import re
import logging
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

load_dotenv()

STYLE_PROMPTS = {
    "essay": """You are a strict academic writing editor.
Your ONLY job is to give bullet-point feedback on the text below.
Do NOT rewrite the text. Do NOT give general writing advice.
Standards: formal tone, clear thesis, structured argumentation.""",
    "fiction": """You are a fiction editor giving targeted revision notes.
Your ONLY job is to give bullet-point feedback on the text below.
Do NOT rewrite the text. Do NOT explain general writing theory.
Standards: narrative voice, pacing, worldbuilding consistency, character clarity.""",
}

# Prompt optimierung
SUB_PROMPTS = {
    "grammar": """TASK: Find grammar and punctuation issues only.
OUTPUT FORMAT:
— use exactly this structure, no numbered lists:
- "problem phrase" → error type → corrected version
Example:
- "it must be remembered that" → passive voice → "note that"
Limit: maximum 8 issues. Ignore style. Ignore word choice.""",
    "style": """TASK: Identify style weaknesses only.
OUTPUT FORMAT:
— use exactly this structure, no numbered lists:
- "relevant phrase" → issue type → one-sentence revision direction
Example:
- "due to the fact that" → wordy phrase → replace with "because"
Limit: maximum 5 issues. Ignore grammar. Ignore content.""",
    "clarity": """TASK: Find sentences or passages that are unclear or hard to follow.
OUTPUT FORMAT:
— use exactly this structure, no numbered lists:
- "unclear passage" → why it's unclear → one specific fix
Example:
- "this ongoing situation" → vague referent, unclear what situation → name the situation explicitly
Limit: maximum 5 issues. Ignore grammar. Ignore style.""",
}


PROMPT_TEMPLATE = """ROLE: {style_context}

WRITING GUIDELINES (use these as your standard):
Only use relevant to your specific task: {context}

---
TASK:
{focus_instruction}

TEXT TO REVIEW (only give feedback on this):
\"\"\"
{question}
\"\"\"

---
RULES:
- Write in en-GB
- Be specific: quote the text directly when referencing it
- Do NOT summarise or praise the text
- Do NOT rewrite full paragraphs
- Stay strictly within the focus area of your task
- If there are no issues to report, write: "No significant issues found."
"""

SUB_PROMPT_QUERIES = {
    "grammar": "grammar punctuation tense agreement comma splice sentence structure",
    "style": "style passive voice register word choice tone verb strength",
    "clarity": "clarity ambiguity sentence readability coherence structure",
}

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma")


def get_db(chroma_path: str, collection_name: str):
    """Initialize and return the ChromaDB collection for the specified style."""
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


def text_normalization(user_text: str):
    """Normalize text by fixing hyphenated line breaks, normalizing line endings, and collapsing spaces with Regex."""
    # Fix hyphenated line breaks
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", user_text)
    # Replace single newlines with spaces
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # Normalize multiple paragraph breaks
    text = re.sub(r"\n{2,}", "\n\n", text)
    # Collapse multiple spaces
    text = re.sub(r"[ ]{2,}", " ", text)

    return text.strip()


def similarity_search_eval(db, query: str, top_k: int = 5):
    """
    Plain similarity search for retrieval evaluation.
    Just tests if the embedding model retrieves the right chunks.
    """
    results = db.similarity_search_with_score(query, k=top_k)

    logger.info(f"++ Eval similarity search ++")
    for doc, score in results:
        logger.info(f"Score: {score:.3f} | Source: {doc.metadata.get('source', '?')}")

    return results  # list of (doc, score)


def similarity_search(db, user_text, collection_name: str, top_k=5):
    """Perform a similarity search on the ChromaDB collection and return the top_k results."""

    all_results = {}

    for focus, focus_query in SUB_PROMPT_QUERIES.items():
        # combine focus with small part of user text
        combined_query = f"{focus_query}\n\n{user_text[:2000]}"
        results = db.similarity_search_with_score(
            combined_query, k=top_k
        )  # hybrid search ?
        # results = db.similarity_search_with_score(user_text[:10000], k=top_k)

        logger.info(
            f"++ Debugging similarity search | Collection '{collection_name}' | Focus '{focus}' ++"
        )
        for doc, score in results:
            logger.info(
                f"Score: {score:.3f} | Source: {doc.metadata.get('source', '?')}"
            )
            logger.debug(f"Content: {doc.page_content[:150]}")

        all_results[focus] = results if results else []

    if not all_results:
        return "No relevant writing guidelines found."

    return all_results


def llm_call(
    user_text_chunks, style_context, context_by_focus: dict, model, all_feedback
):
    """Make LLM calls for each chunk of user text and collect feedback for grammar, style, and clarity."""
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    for i, user_text_chunk in enumerate(user_text_chunks):
        logger.debug(
            f"6.{i+1} Invoking LLM with chunk {i+1}/{len(user_text_chunks)}. Chunklength: {len(user_text_chunk.split())} words."
        )

        for section, focus in SUB_PROMPTS.items():

            # now each sub prompts gets its own context
            context_text = context_by_focus.get(section, "")

            prompt = prompt_template.format(
                style_context=style_context,
                focus_instruction=focus,
                context=context_text,
                question=user_text_chunk,
            )
            logger.debug(f"7.{i+1} Running sub-prompt: {section}")

            try:
                response = model.invoke(prompt)
                all_feedback[section].append(response)
            except Exception as e:
                logger.error(f"LLM invocation failed: {e}")
                raise

    return all_feedback


def format_feedback(sections: dict) -> str:
    """Format the collected feedback into a structured string output."""
    formatted_sections = {}
    for section, responses in sections.items():
        if len(responses) == 1:
            joined = responses[0].strip()
        else:
            # Multi-chunk: just join with a divider, no "Chunk N" label
            joined = "\n\n---\n\n".join(r.strip() for r in responses)
        formatted_sections[section] = joined

    return (
        f"### Grammar\n{formatted_sections['grammar']}\n\n"
        f"### Style\n{formatted_sections['style']}\n\n"
        f"### Clarity\n{formatted_sections['clarity']}"
    )


def query_rag(user_text: str, style: str = "essay", return_dict: bool = False) -> str:
    """Main function to handle the RAG process for writing feedback."""

    user_text = text_normalization(user_text)
    # Additional Logging to check formating of text chunks. In case linebraks or hyphons aren't properly formated
    logger.debug(f"1. Normalized user text {user_text[0:200]}")

    if not user_text:
        return "Please provide some text to get feedback on."

    if len(user_text) > 50000:
        return (
            "Text is too long! It's over 50'000 characters. That's almost 10'000 words. Do you want to wait for an hour?"
            "Please split it into smaller sections."
        )

    logger.debug("2. Input validation passed. Proceeding with RAG process.")
    collection_name = style if style in STYLE_PROMPTS else "essay"
    style_context = STYLE_PROMPTS.get(collection_name, STYLE_PROMPTS["essay"])
    try:
        db = get_db(CHROMA_PATH, collection_name)
    except (FileNotFoundError, ValueError) as e:
        error_msg = f"Database error: {e}"
        if return_dict:
            raise RuntimeError(error_msg)
        return error_msg

    start = time.time()
    top_k = int(os.getenv("TOP_K", "3"))
    results = similarity_search(
        db, user_text, collection_name=collection_name, top_k=top_k
    )
    context_text = {
        focus: "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        for focus, results in results.items()
    }

    end = time.time()
    logger.debug(
        f"3. Completed similarity search for context for LLM in {end - start:.2f} seconds."
    )

    model = OllamaLLM(
        model=os.getenv("LOCAL_MODEL", "qwen3.5:4b"),
        base_url="http://127.0.0.1:11434",
        temperature=float(os.getenv("TEMPERATURE", "0.7")),
    )
    logger.debug(f"4. LLM model initialized: {os.getenv('LOCAL_MODEL', 'qwen3.5:4b')}")

    # Split at ~5000 characters to Improve quality.

    user_text_split = int(os.getenv("USER_TEXT_SPLIT", "5000"))
    user_text_chunks = embeddings.chunk_user_prompt(
        user_text, chunk_size=user_text_split, chunk_overlap=int(user_text_split * 0.2)
    )
    logger.debug(f"5. User text chunking completed.")

    all_feedback = {"grammar": [], "style": [], "clarity": []}

    start = time.time()
    all_feedback = llm_call(
        user_text_chunks, style_context, context_text, model, all_feedback
    )
    end = time.time()
    logger.debug(f"8. LLM response time: {end - start:.2f} seconds")
    if return_dict:
        return all_feedback
    return format_feedback(all_feedback)
