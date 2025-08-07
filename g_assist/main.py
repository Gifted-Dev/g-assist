import typer
from google.generativeai import protos
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from typing import Optional
import os

from .tools import execute_shell_command
from .models.base import BaseModel
from .models.gemini import GeminiModel

console = Console()
app = typer.Typer()

def extract_text_from_response(response) -> str:
    """Safely extract concatenated text from a Gemini response's parts."""
    try:
        cand = response.candidates[0]
        parts = getattr(cand.content, "parts", []) or []
        texts = []
        for p in parts:
            t = getattr(p, "text", None)
            if t:
                texts.append(t)
        return "\n".join(texts).strip()
    except Exception:
        return ""

shell_tool = protos.Tool(
    function_declarations=[
        protos.FunctionDeclaration(
            name="execute_shell_command",
            description="Executes a command in the system's shell and returns the output.",
            parameters=protos.Schema(
                type=protos.Type.OBJECT,
                properties={
                    "command": protos.Schema(type=protos.Type.STRING, description="The command to execute.")
                },
                required=["command"],
            ),
        )
    ]
)



def start_chat(model: BaseModel):
    """Starts an interactive chat session with the AI."""
    
    chat = model.start_chat(
        history=[], enable_automatic_function_calling=False
    )  # Initializing a new chat object with manual tool handling
    
    console.print("[bold green]Welcome to G-Assist! [/bold green]Type 'exit' or 'quit' to end the chat.")
    while True:
        prompt = console.input("[bold cyan]You: [/bold cyan]")

        # --------------- Check if the user's input, after cleaning it up, is an exact match for an exit command.-----------------#
        
        if prompt.strip().lower() in ["exit", "quit"]:
            console.print("[bold green]G-assist: [/bold green]", end="")
            console.print("[bold yellow]Thanks for coming around. Bye![/bold yellow]")
            raise typer.Exit(code=0)

        # --------------------------------------------------------------------------------------------------------#
        
        
        with console.status("[bold green]Thinking...[/bold green]"):
            try:
                response = chat.send_message(prompt)
                text = extract_text_from_response(response)
                if text:
                    to_print = Markdown(text)
                else:
                    to_print = "[bold red]:warning: The model finished its work but did not provide a final text response.[/bold red]"
            except ValueError:
                # This can happen if the model returns a function call and not text.
                to_print = "[bold red]:warning: The model finished its work but did not provide a final text response.[/bold red]"
            except Exception as e:
                response_text = f"**Error:** An error occurred while communicating with G-Assist.\n\n*Details: {e}*"
                to_print = Markdown(response_text)
        console.print("[bold green]G-assist: [/bold green]", end="")
        console.print(to_print)


@app.command()
def main(
    prompt: Optional[str] = typer.Argument(
        None, help="The prompt to send to the AI for a single response."
    ),
):
    """
    G-Assist: Your CLI-based AI assistant.
    Run without a prompt to start an interactive chat session.
    """
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        console.print("[bold red]Error: GOOGLE_API_KEY not found in .env file.[/bold red]")
        raise typer.Exit(code=1)

    # This is the only place we need to import genai directly now
    import google.generativeai as genai
    genai.configure(api_key=api_key)

    system_instruction = (
    "You are G-Assist, a highly capable agent, code assistant, and command-line tool designed to support software developers. "
    "Your core responsibilities include: explaining and breaking down complex codebases, resolving terminal errors and command-line issues, "
    "performing developer-oriented tasks via available tools, and assisting interactively as a programming agent. "
    "You operate primarily in a technical capacity, helping users understand code behavior, debug problems, and automate workflows. "
    "You have access to tools and must decide whether to invoke a tool, and which command to execute when appropriate. "
    "After receiving tool output, you MUST always interpret and use it to produce a clear, human-readable response that directly addresses the user's question or problem. "
    "NEVER stop at tool output alone—ALWAYS follow up with a final textual explanation or summary. "
    "Once you've completed a task or answered a question, offer to provide further clarification or a deeper explanation."
    )


    # Create the model *after* configuring the API key
    model = GeminiModel(
        model_name='gemini-2.5-pro',
        tools=[shell_tool],
        system_instruction=system_instruction
    )

    if prompt:
        # Handle single-shot questions
        with console.status("[bold green]Thinking...[/bold green]"):
            try:
                response = model.generate_content(prompt)
                text = extract_text_from_response(response)
                if text:
                    to_print = Markdown(text)
                else:
                    to_print = "[bold red]:warning: The model did not provide a final text response.[/bold red]"
            except ValueError:
                # This can happen if the model returns a function call and not text.
                to_print = "[bold red]:warning: The model did not provide a final text response.[/bold red]"
            except Exception as e:
                response_text = (
                    f"**Error:** An error occurred while communicating with G-Assist.\n\n"
                    f"*Details: {e}*"
                )
                to_print = Markdown(response_text)
        console.print("[bold green]G-assist: [/bold green]", end="")
        console.print(to_print)
    else:
        start_chat(model)
