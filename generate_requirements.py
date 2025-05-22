import os
import re
import pkg_resources
from pathlib import Path

def extract_imports(file_path):
    """Extracts all imports from a Python file."""
    imports = set()
    with open(file_path, "r") as file:
        for line in file:
            # Match standard import statements
            match = re.match(r"^\s*(?:import|from)\s+([\w\.]+)", line)
            if match:
                imports.add(match.group(1).split('.')[0])  # Get the top-level package
    return imports

def get_installed_version(package_name):
    """Gets the installed version of a package."""
    try:
        return pkg_resources.get_distribution(package_name).version
    except pkg_resources.DistributionNotFound:
        return None

def generate_requirements(workspace_path, output_file="requirements.txt"):
    """Generates a requirements.txt file based on imports in Python files."""
    all_imports = set()
    workspace_path = Path(workspace_path)

    # Walk through all files in the workspace
    for root, _, files in os.walk(workspace_path):
        for file in files:
            if file.endswith(".py"):  # Only process Python files
                file_path = Path(root) / file
                all_imports.update(extract_imports(file_path))

    # Get versions for all imports
    requirements = []
    for package in sorted(all_imports)[1:]:
        version = get_installed_version(package)
        if version:
            requirements.append(f"{package}=={version}")
        else:
            requirements.append(package)  # Add without version if not installed

    # Write to requirements.txt
    with open(output_file, "w") as req_file:
        req_file.write("\n".join(requirements))
    print(f"Requirements written to {output_file}")

if __name__ == "__main__":
    workspace_path = "."  # Adjust this to the root of your workspace if needed
    generate_requirements(workspace_path)