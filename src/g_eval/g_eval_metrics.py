
## ToDo:
    # Split into 3 Steps. Grammer, Style and Clarity each get a score of 1-5.
    # So we rate each individual LLM call seperately.
    # Additionally we have a Faithfullnes Metric. This determines if the Collected Rag Contexts are Faithfull to the User Prompt. 

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase
from deepeval.test_case import LLMTestCaseParams, LLMTestCase
from deepeval.models import OllamaModel
# from dotenv import load_dotenv
# load_dotenv()

ollama_model = OllamaModel(
    model="llama3.1:8b"
)

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

grammar_metric = GEval(
    model=ollama_model,
    name="Grammar Accuracy",
    criteria=grammar_criteria,
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
)


style_criteria = """
Style Feedback Quality (1–5):

Evaluate how well the feedback improves the writing style
of the user's text.

Consider:
1. Tone – Does the feedback improve formality or appropriateness?
2. Flow – Does it improve sentence smoothness and readability?
3. Word Choice – Does it suggest clearer or more natural phrasing?
4. Readability – Does it help make sentences more engaging or fluent?

Scoring:
1 – Style feedback is incorrect, irrelevant, or harmful.
2 – Some useful suggestions but many weak or unnecessary ones.
3 – Generally helpful but limited in depth or coverage.
4 – Useful and effective style improvements.
5 – Highly effective style feedback that significantly improves fluency and readability.
"""

style_metric = GEval(
    model=ollama_model,
    name="Style Accuracy",
    criteria=style_criteria,
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
)

clarity_criteria = """
Clarity Feedback Quality (1–5):

Evaluate how well the feedback improves the clarity
and understandability of the user's text.

Consider:
1. Comprehensibility – Does the feedback make ideas easier to understand?
2. Structure – Does it improve sentence or paragraph organization?
3. Ambiguity – Does it reduce confusion or unclear meaning?
4. Simplicity – Does it suggest clearer ways to express ideas?

Scoring:
1 – Feedback does not improve clarity or introduces confusion.
2 – Some helpful suggestions but many unclear or weak ones.
3 – Generally improves clarity but misses important issues.
4 – Effectively improves understanding of the text.
5 – Significantly enhances clarity and removes confusion throughout.
"""

clarity_metric = GEval(
    model=ollama_model,
    name="Clarity Accuracy",
    criteria=clarity_criteria,
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
)


def evaluate_grammar(user_text, grammar_feedback):
    test_case = LLMTestCase(
        input=user_text,
        actual_output=grammar_feedback,
    )
    grammar_metric.measure(test_case)

    print(f"Grammar Score: {grammar_metric.score}")
    print(f"Reason: {grammar_metric.reason}")


def evaluate_style(user_text, style_feedback):
    test_case = LLMTestCase(
        input=user_text,
        actual_output=style_feedback,
    )
    style_metric.measure(test_case)

    print(f"Style Score: {style_metric.score}")
    print(f"Reason: {style_metric.reason}")

def evaluate_clarity(user_text, clarity_feedback):
    test_case = LLMTestCase(
        input=user_text,
        actual_output=clarity_feedback,
    )

    clarity_metric.measure(test_case)
    print(f"Clarity Score: {clarity_metric.score}")
    print(f"Reason: {clarity_metric.reason}")