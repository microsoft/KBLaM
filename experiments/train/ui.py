from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.theme import Theme

# Create a custom theme for Rich
custom_theme = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "critical": "bold white on red",
    }
)

# Create a Rich console with the custom theme
console = Console(theme=custom_theme)

def create_custom_progress_bar(
    console: Console = None,  # type: ignore
    color: str = "cyan",
    show_time: bool = True,
    show_spinner: bool = True,
    spinner_style: str = "dots",
    disable=False,
) -> Progress:
    """
    Creates a customizable progress bar using the Rich library.

    This function configures and returns a Rich Progress object with various
    columns for displaying progress, such as a spinner, a description, a bar,
    task progress percentage, loss value, and remaining time.

    Args:
        console (Console, optional): A Rich Console object. If None, a new one is created. Defaults to None.
        color (str, optional): The color of the progress bar and spinner. Defaults to "cyan".
        show_time (bool, optional): Whether to display the time remaining. Defaults to True.
        show_spinner (bool, optional): Whether to display a spinner. Defaults to True.
        spinner_style (str, optional): The style of the spinner. Defaults to "dots".
        disable (bool, optional): If True, the progress bar will not be displayed. Defaults to False.

    Returns:
        Progress: A configured Rich Progress object.
    """
    if console is None:
        console = Console()
    columns = []

    if show_spinner:
        columns.append(SpinnerColumn(spinner_name=spinner_style, style=color))

    columns.extend(
        [
            TextColumn("[bold blue]{task.description}", justify="right"),
            BarColumn(bar_width=None, style=color, complete_style=f"bold {color}"),
            TaskProgressColumn(),
            TextColumn("[bold yellow]Loss: {task.fields[loss]:.4f}", justify="right"),
        ]
    )

    if show_time:
        columns.append(TimeRemainingColumn())

    progress = Progress(*columns, console=console, expand=True, disable=disable)
    return progress
