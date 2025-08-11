import os
import sys
import pandas as pd

def extract_vasp_structure_types(directory):
    """
    Scans a directory (recursively) for files ending with '.vasp' and extracts
    the structure ID and structure type from each filename.

    Assumes that:
        - The structure ID is the filename without the '.vasp' extension.
        - The structure type is the last underscore-separated token in the filename
          before the '.vasp' extension.

    Example:
        'abc_123_TYPE.vasp' → id='abc_123_TYPE', structure_type='TYPE'

    Args:
        directory (str): Path to the directory to scan.

    Returns:
        pd.DataFrame: DataFrame with columns ['id', 'structure_type'].
    """
    records = []
    stack = [directory]  # stack for recursion

    while stack:
        current_dir = stack.pop()
        try:
            with os.scandir(current_dir) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(entry.path)
                    elif entry.name.endswith(".vasp"):
                        file_id = entry.name.rsplit(".", 1)[0]
                        structure_type = file_id.split("_")[-1]
                        records.append((file_id, structure_type))
        except PermissionError:
            continue  # skip directories without access permissions

    return pd.DataFrame(records, columns=["id", "structure_type"])

def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {os.path.basename(__file__)} <directory_path>")
        sys.exit(1)

    directory_path = sys.argv[1]

    if not os.path.isdir(directory_path):
        print(f"Error: '{directory_path}' is not a valid directory.")
        sys.exit(1)

    df = extract_vasp_structure_types(directory_path)
    output_file = os.path.join(directory_path, "delafossite_structure_type.csv")
    df.to_csv(output_file, index=False)

    print(f"CSV file saved to: {output_file}")

if __name__ == "__main__":
    main()
