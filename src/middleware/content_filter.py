import re
from typing import List, Optional, Any, Dict

from langchain.agents.middleware import AgentMiddleware, hook_config

# implements a before_agent hook that blocks requests containing prompt-injection patterns.
class ContentFilterMiddleware(AgentMiddleware):


    def __init__(self, logger=None, extra_blocklist: Optional[List[str]] = None):
        super().__init__()
        self.logger = logger
        self.blocklist = [
            r"ignore.*?previous.*(instructions|context|prompts)",
            r"disregard.*(previous|above).*(instructions|context|prompts)",
            r"forget.*?\b(previous|above)\b",
            r"override.*system instructions",
            r"follow.*these instructions",
            r"you are now.*?\bassistant\b",
            r"disobey",
            r"bypass.*security|bypass the",
            r"insert code to",
            r"execute the following",
            r"run this code",
        ]

        if extra_blocklist:
            self.blocklist.extend(extra_blocklist)

        self._compiled = [re.compile(pat, re.IGNORECASE) for pat in self.blocklist]

    def _matches_block(self, text: str) -> List[str]:
        matches = []
        for cre in self._compiled:
            if cre.search(text):
                matches.append(cre.pattern)
        return matches

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: dict, runtime: Any) -> Dict[str, Any] | None:
        # Expect state (dict) to contain a messages list; get first human message
        messages = state.get("messages")
        if not messages:
            return None

        first_message = messages[0]
        # supports both dict-like message or small helper objects
        content = None
        if isinstance(first_message, dict):
            # doc examples use `type` and `content`
            if first_message.get("type") != "human":
                return None
            content = first_message.get("content", "")
        else:
            # fall back: try attribute access
            if getattr(first_message, "type", None) != "human":
                return None
            content = getattr(first_message, "content", "")

        if not content or not content.strip():
            return None

        matched = self._matches_block(content)
        if matched:
            if self.logger:
                self.logger.warning(
                    "ContentFilterMiddleware: blocked input matching patterns: %s",
                    matched,
                )
            return {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "I cannot process requests containing unsafe or manipulative instructions. Please rephrase your request.",
                    }
                ],
                "jump_to": "end",
            }

        if len(content) > 20000 and content.count("\n") < 2:
            if self.logger:
                self.logger.warning("ContentFilterMiddleware: blocked long single-line input.")
            return {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "Input rejected: suspicious long single-line payload.",
                    }
                ],
                "jump_to": "end",
            }

        return None

    # Convenience helper for non-agent code paths (keeps backwards-compatibility)
    def check_text(self, text: str) -> bool:
        """Return True if text passes checks, False if blocked."""
        if not text or not text.strip():
            return True
        matched = self._matches_block(text)
        if matched:
            if self.logger:
                self.logger.warning("ContentFilterMiddleware: blocked patterns %s", matched)
            return False
        if len(text) > 20000 and text.count("\n") < 2:
            if self.logger:
                self.logger.warning("ContentFilterMiddleware: blocked long single-line input.")
            return False
        return True
