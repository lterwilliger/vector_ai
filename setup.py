from setuptools import setup, find_packages

setup(
    name="vector_ai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "pytest",
        "sentence-transformers",
    ],
    python_requires=">=3.8",
) 