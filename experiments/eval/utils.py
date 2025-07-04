import json
from pathlib import Path
from typing import Any

def write_to_json(
    data: Any, filepath: str, indent: int = 4, encoding: str = "utf-8"
) -> bool:
    """Writes a dictionary to a JSON file with specified formatting.

    This function serializes a Python dictionary to a JSON file with error handling.
    It allows for custom indentation and encoding.

    Args:
        data (Any): The dictionary or other serializable object to write to the file.
        filepath (str): The path to the output JSON file.
        indent (int, optional): The number of spaces for JSON indentation. Defaults to 4.
        encoding (str, optional): The file encoding. Defaults to 'utf-8'.

    Returns:
        bool: True if the file was written successfully, although the function does not explicitly return a value.
    """

    try:
        # Convert string path to Path object
        file_path = Path(filepath)

        # Write the JSON file
        with open(file_path, "w", encoding=encoding) as f:
            json.dump(
                data,
                f,
                indent=indent,
                sort_keys=True,  # For consistent output
                default=str,  # Handle non-serializable objects by converting to string
            )

    except Exception as e:
        print(f"Error writing JSON file: {str(e)}")
