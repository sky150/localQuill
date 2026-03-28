import src.g_eval.g_eval_metrics as g_eval_metrics
from src.model_query import query_rag


def run_full_evaluation(user_prompt):
    """Run the full evaluation process for grammar, style, and clarity."""
    feedback = query_rag(
        user_prompt,
        style="essay", # fiction
        return_dict=True
    )


    grammar_feedback = "\n".join(feedback["grammar"])
    style_feedback = "\n".join(feedback["style"])
    clarity_feedback = "\n".join(feedback["clarity"])

    print("\n--- Running Grammar Evaluation ---")

    g_eval_metrics.evaluate_grammar(
        user_prompt,
        grammar_feedback
    )

    print("\n--- Running Style Evaluation ---")

    g_eval_metrics.evaluate_style(
        user_prompt,
        style_feedback
    )

    print("\n--- Running Clarity Evaluation ---")

    g_eval_metrics.evaluate_clarity(
        user_prompt,
        clarity_feedback
    )



if __name__ == "__main__":

    test_user_prompt = """It cannot be denied that tourism is an economic force \n
and that over the past decades it has gone from strength
to strength. Both texts give us some insight into one
important link for the industry, and that is the internet.
The first text makes it clear that travel, whether doing
so on a restricted budget or with no such limitations, is
facilitated with the amount of online material to choose
from. This may, of course, include such things as travel
vlogs, such as the ones produced by the author of the
second article. This second author writes from a personal
viewpoint and her perspective confirms that the world
wide web is indeed a rich resource.\n
\n
Reading between the lines, the travel vlogger appears
to show her passion and extraordinary enthusiasm for
travel. This is somewhat missing in the first text where
the process of identifying where to go and what to do is
made to sound so much more mechanical. The vlogger
thinks beyond what is in front of them and considers
deeply what the experiences bring her in terms of the
things she thinks are most important and valuable. The
writer of the first article almost dismisses this aspect by
focusing on the simplicity of taking a trip.
Although I have not travelled extensively, I can relate
to the vlogger and feel any tourism company would
benefit from the clear delight behind each experience in
promoting their company."""

    run_full_evaluation(test_user_prompt)