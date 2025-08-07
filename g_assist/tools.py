import subprocess


def execute_shell_command(command: str) -> str:
    """Executes a shell command and returns its output or error.

    Adds a timeout to prevent hanging processes and returns combined stderr/stdout
    for clearer context.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds."

    output = (result.stdout or "").strip()
    error = (result.stderr or "").strip()

    if result.returncode == 0:
        return output
    else:
        # Return both outputs for context when a command fails
        combined = f"{output}\n{error}".strip()
        return f"Error: {combined}" if combined else "Error: Command failed with no output."
