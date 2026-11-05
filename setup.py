"""Install the existing src/greenforge_ai package without changing its layout."""

from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent


def get_requirements():
    """Exclude headings, empty lines and editable-install markers."""
    return [
        line.strip()
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith(("#", "-e "))
    ]


setup(
    name="greenforge-ai",
    version="0.0.1",
    author="Reza Mansouri",
    author_email="rezamansouri219@gmail.com",
    description="AI-powered sustainable manufacturing optimization.",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=get_requirements(),
    python_requires=">=3.12",
)
