from setuptools import setup, find_packages

setup(
    name="memory-layer",
    version="0.1.0",
    description="A biologically-inspired memory system for AI. "
                "Persistent, evolving, local memory that any AI can plug into.",
    author="Memory Layer",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pydantic>=2.0.0",
        "numpy>=1.24.0",
        "sentence-transformers>=2.2.0",
        "faiss-cpu>=1.7.0",
        "fastapi>=0.100.0",
        "uvicorn[standard]>=0.23.0",
    ],
    entry_points={
        "console_scripts": [
            "memory-layer=run:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
    ],
)
