from setuptools import find_packages, setup

def get_requirements(file_path):
    with open(file_path, "r") as file:
        requirements = file.read().splitlines()
        
    if "-e ." in requirements:
        requirements.remove("-e .")
        
    return requirements

setup(
    name="greenforge-ai",
    version="0.0.1",
    author="Reza Mansouri",
    author_email="rezamansouri219@gmail.com",
    description="AI-powered sustainable manufacturing optimization.",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=get_requirements("requirements.txt"),
    python_requires=">=3.12",
)