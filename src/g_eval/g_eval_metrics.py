
## ToDo:
    # Split into 3 Steps. Grammer, Style and Clarity each get a score of 1-5.
    # So we rate each individual LLM call seperately.
    # Additionally we have a Faithfullnes Metric. This determines if the Collected Rag Contexts are Faithfull to the User Prompt. 

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase
from deepeval.models import OllamaModel
# from dotenv import load_dotenv
# load_dotenv()

grammar_criteria = """
Grammar Feedback Quality (1–5):

Evaluate whether the grammar feedback correctly identifies
and explains grammar mistakes in the user's text.

Consider:
1. Accuracy – Are grammar errors correctly identified?
2. Coverage – Are important grammar errors detected?
3. Precision – Are false grammar errors avoided?
4. Clarity – Are explanations understandable?

Scoring:
1 – Mostly incorrect or misleading feedback.
2 – Some correct grammar observations but unreliable.
3 – Generally correct but incomplete.
4 – Accurate and useful grammar feedback.
5 – Highly accurate, comprehensive, and precise.
"""

ollama_model = OllamaModel(
    model="llama3.1:8b"
)

grammar_metric = GEval(
    model=ollama_model,
    name="Grammar Accuracy",

    criteria=grammar_criteria,

    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
)

from deepeval.test_case import LLMTestCase

def evaluate_grammar(user_text, grammar_feedback):

    test_case = LLMTestCase(
        input=user_text,
        actual_output=grammar_feedback,
    )

    grammar_metric.measure(test_case)

    print(f"Grammar Score: {grammar_metric.score}")
    print(f"Reason: {grammar_metric.reason}")
