import importlib
import os
import sys
import pkg_resources
import subprocess
import importlib.util

node_list = [
    "moondream_script",
    "description",
    "simple_prompt",
    "joytag",
]

# Define the check_requirements_installed function here or import it
def check_requirements_installed(requirements_path):
    with open(requirements_path, 'r') as f:
        requirements = [pkg_resources.Requirement.parse(line.strip()) for line in f if line.strip()]

    installed_packages = {pkg.key: pkg for pkg in pkg_resources.working_set}
    missing_packages = []
    for requirement in requirements:
        if requirement.key not in installed_packages or not installed_packages[requirement.key] in requirement:
            missing_packages.append(str(requirement))

    if missing_packages:
        print(f"Missing or outdated packages: {', '.join(missing_packages)}")
        print("Installing/Updating missing packages...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing_packages])
    else:
        print("All packages from requirements.txt are installed and up to date.")


requirements_path  = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")
check_requirements_installed(requirements_path)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for module_name in node_list:
    imported_module = importlib.import_module(f".nodes.{module_name}", __name__)

    NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **imported_module.NODE_CLASS_MAPPINGS}
    NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **imported_module.NODE_DISPLAY_NAME_MAPPINGS}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

