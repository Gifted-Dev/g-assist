import google.generativeai as genai
from .base import BaseModel, BaseChatSession
from typing import Any, List, Optional
from google.generativeai import protos
from ..tools import execute_shell_command

tool_registry = {
    "execute_shell_command": execute_shell_command
}
    
class GeminiChatSession(BaseChatSession):
    """Concrete implementation of BaseChatSession for the Gemini model."""

    def __init__(self, model: "GeminiModel", history: List[Any], enable_automatic_function_calling: bool):
        self.model = model
        # This holds the actual chat session object from the google-generativeai library
        self._chat_session = model._model.start_chat(
            history=history, enable_automatic_function_calling=enable_automatic_function_calling
        )

    def send_message(self, prompt: str, **kwargs: Any) -> Any:
        """Sends a message to the Gemini API and handles function-calling tool execution until a final response is produced."""
        # Send the user's message first
        response = self._chat_session.send_message(prompt, **kwargs)

        while True:
            # Try to detect a function call from the assistant
            try:
                part = response.candidates[0].content.parts[0]
                function_call = getattr(part, "function_call", None)
                if not (function_call and function_call.name):
                    break  # No tool call; we have a final answer
            except (AttributeError, IndexError):
                break  # Malformed or no function call; treat as final

            # Execute the requested tool
            function_name = function_call.name
            if function_name not in tool_registry:
                raise ValueError(f"Tool {function_name} not found in registry.")

            tool_function = tool_registry[function_name]
            args = {key: value for key, value in function_call.args.items()}
            result = tool_function(**args)

            # Send the tool's function response back into the chat to let the model continue
            tool_response_content = protos.Content(
                parts=[
                    protos.Part(
                        function_response=protos.FunctionResponse(
                            name=function_name,
                            response={"output": result},
                        )
                    )
                ],
                role="tool",
            )
            response = self._chat_session.send_message(tool_response_content)

        return response


class GeminiModel(BaseModel):
    """Concrete implementation of BaseModel for the Gemini model."""

    def __init__(self, model_name: str, tools: List[Any], system_instruction: Optional[str] = None):
        # This holds the actual GenerativeModel object from the google-generativeai library
        self._model = genai.GenerativeModel(
            model_name=model_name,
            tools=tools,
            system_instruction=system_instruction
        )

    def start_chat(
        self, history: Optional[List[Any]] = None, *, enable_automatic_function_calling: bool = False
    ) -> GeminiChatSession:
        """Starts a Gemini chat session."""
        if history is None:
            history = []
        return GeminiChatSession(self, history=history, enable_automatic_function_calling=enable_automatic_function_calling)

    def generate_content(self, prompt: str, **kwargs: Any) -> Any:
        """Generates content for a single, non-chat request."""
        # This is the manual agent loop for single-shot commands.
        history = [protos.Content(parts=[protos.Part(text=prompt)], role="user")]

        while True:
            response = self._model.generate_content(history, **kwargs)

            try:
                part = response.candidates[0].content.parts[0]
                function_call = getattr(part, "function_call", None)
                if not (function_call and function_call.name):
                    # This is a final text response or a malformed/empty function call, so we're done.
                    break
            except (AttributeError, IndexError):
                # Something went wrong or no function call, so we're done.
                break

            # --- Function Call Execution ---
            function_name = function_call.name
            if function_name not in tool_registry:
                raise ValueError(f"Tool {function_name} not found in registry.")

            tool_function = tool_registry[function_name]
            args = {key: value for key, value in function_call.args.items()}
            result = tool_function(**args)

            # --- Send the result back to the model ---
            history.append(response.candidates[0].content)  # Add the AI's request to history
            history.append(
                protos.Content(
                    parts=[
                        protos.Part(function_response=protos.FunctionResponse(name=function_name, response={"output": result}))
                    ],
                    role="tool",
                )
            )
            # The loop will now run again, sending the tool's output to the AI.

        return response