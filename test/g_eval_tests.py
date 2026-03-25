import src.g_eval.g_eval_metrics as g_eval_metrics


def test_grammer_evaluation(test_user_prompt):

    test_llm_grammer_feedback = """
    Grammar
Chunk 1:
Here is the feedback on grammar errors:
1. "It cannot be denied that tourism is an economic force and that over the past decades it has gone from strength to strength."
The phrase "and that" is a bit informal for academic writing. Consider rephrasing as: "It cannot be denied that tourism is an economic force, which has grown significantly over the past decades."
2. "Both texts give us some insight into one important link for the industry, and that is the internet."
The phrase "and that" is used again; consider rephrasing as: "...give us some insight into one important link for the industry, namely the internet."
3. "The first text makes it clear that travel, whether doing so on a restricted budget or with no such limitations, is facilitated with the amount of online material to choose from."
The phrase "with the amount of" is a bit informal; consider rephrasing as: "...is facilitated by the abundance of online material available."
4. "This may, of course, include such things as travel vlogs, such as the ones produced by the author of the second article."
Consider adding a comma after "vlogs" to improve sentence structure.
5. "The writer of the first article almost dismisses this aspect by focusing on the simplicity of taking a trip."
The phrase "almost dismisses" is a bit vague; consider rephrasing as: "...by downplaying the importance of this aspect in favor of emphasizing the simplicity of travel."
6. "Although I have not travelled extensively, I can relate to the vlogger and feel any tourism company would benefit from the clear delight behind each experience in promoting their company."
Consider adding a transition word or phrase (e.g., "Furthermore," "In addition") to connect this sentence to the rest of the text.

These are just some suggestions for improvement.
"""

    g_eval_metrics.evaluate_grammar(test_user_prompt, test_llm_grammer_feedback)


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

    test_grammer_evaluation(test_user_prompt)



