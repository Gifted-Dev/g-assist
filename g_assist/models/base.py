from abc import ABC, abstractmethod
from typing import Any, List


class BaseChatSession(ABC):
    """
    Abstract base class for a chat session.
    Defines the standard interface for a stateful conversation.
    """

    @abstractmethod
    def send_message(self, prompt: str, **kwargs: Any) -> Any:
        """Sends a message to the model and gets a response."""
        pass


class BaseModel(ABC):
    """
    Abstract base class for a generative model.
    Defines the standard interface for creating and interacting with a model.
    """

    @abstractmethod
    def start_chat(
        self, history: List[Any], *, enable_automatic_function_calling: bool = False
    ) -> BaseChatSession:
        """Starts a new chat session and returns a session object."""
        pass

    @abstractmethod
    def generate_content(self, prompt: str, **kwargs: Any) -> Any:
        """Generates content for a single, non-chat request."""
        pass