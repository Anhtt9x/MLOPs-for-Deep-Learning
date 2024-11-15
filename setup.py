from setuptools import setup, find_packages

with open('README.md', 'r', encoding="utf-8") as f:
    long_description=f.read()


__version__ = "0.0.0"

REPO_NAME = "MLOPs-for-Deep-Learning"
AUTHOR_USER_NAME = "Anhtt9x"
SRC_REPO = "CnnClassifier"
AUTHOR_EMAIL = "anhtt454598@gmail.com"

setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for CNN app",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    packages=find_packages(where="src"),
    package_dir={"":"src"},
)
