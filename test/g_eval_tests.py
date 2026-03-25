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


def test_style_evaluation(test_user_prompt):

    test_llm_style_feedback = """
Style
Chunk 1:
Here's my feedback on your writing style:
Transitions between sentences are sometimes abrupt: For example, "It cannot be denied that tourism is an economic force... Both texts give us some insight into one important link for the industry..." The transition from a general statement about tourism to a specific reference to two texts feels sudden. Consider adding a sentence or phrase to connect these ideas more smoothly.
Use of phrases like "of course" can weaken argumentation: In the second paragraph, you write: "This may, of course, include such things as travel vlogs..." The word "of course" implies that this is an obvious point, which may undermine your case. Instead, consider rephrasing to make it clear why travel vlogs are relevant to the discussion.
Use more precise language: In the sentence: "The first text makes it clear that travel, whether doing so on a restricted budget or with no such limitations, is facilitated with the amount of online material to choose from," you use the phrase "facilitated with." Consider rephrasing to make this point more clearly. For example: "The first text highlights how the abundance of online resources facilitates travel."
Use active voice instead of passive: In the sentence: "This second author writes from a personal viewpoint and her perspective confirms that the world wide web is indeed a rich resource," consider rephrasing to use active voice, e.g., "The second author presents a personal viewpoint that confirms..."
Transitions between paragraphs could be smoother: The fourth paragraph feels like an abrupt shift in topic. Consider adding a sentence or phrase to connect this idea more clearly to the previous discussion.
Overuse of quotes from other texts is not necessary: While it's good practice to reference primary sources, in this case, the quotation from "The Theory of Moral Sentiments" is unnecessary and detracts from your own argument.
"""

    g_eval_metrics.evaluate_style(test_user_prompt, test_llm_style_feedback)

def test_clarity_evaluation(test_user_prompt):

    test_llm_clarity_feedback = """
Clarity
Chunk 1:
Here's my feedback on clarity:
Suggestions for improvement
Start with a clearer thesis statement: The first paragraph is a good attempt to orient readers, but it could be more concise and clear about what the paper will argue. Consider rephrasing: "This essay argues that...".
Use transitional phrases: To improve flow between ideas, use transitional phrases such as "In addition", "Moreover", or "Furthermore". For example, after discussing the first text, you write "However" to contrast with the second text. This works well.
Be more specific when referencing texts: In some places, it's unclear which article is being referred to. For instance, "This may include such things as travel vlogs, such as the ones produced by the author of the second article." Try rephrasing: "The second article, written from a personal viewpoint, highlights the importance of online resources in travel planning."
Use active voice: Some sentences are written in passive voice (e.g., "the process of identifying where to go and what to do is made to sound so much more mechanical"). Consider rephrasing: "The first text makes the process of identifying...sound...".
Avoid vague terms: Phrases like "it cannot be denied" or "it's clear" are vague. Try using specific language to support your claims (e.g., "the data shows that tourism has grown significantly over the past decades").
Specific feedback
In the first paragraph, consider adding more detail about what you mean by "one important link for the industry".
When discussing travel vlogs, be more precise about how they facilitate travel planning.
In the sentence: "This second author writes from a personal viewpoint and her perspective confirms that the world wide web is indeed a rich resource." Consider rephrasing: "The second article highlights the value of online resources in travel planning through its personal account."
When comparing the two texts, be more specific about what you mean by "passion" and "mechanical".
In the last paragraph, consider adding more detail about how tourism companies can benefit from promoting their company.
Overall, your text is clear and well-structured. With some refinements to clarity, it will become even stronger!
"""

    g_eval_metrics.evaluate_clarity(test_user_prompt, test_llm_clarity_feedback)

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
    test_style_evaluation(test_user_prompt)
    test_clarity_evaluation(test_user_prompt)



