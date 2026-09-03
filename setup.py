"""Packaging for Page-EntroKV."""

from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent


def read(*parts: str) -> str:
    return (ROOT.joinpath(*parts)).read_text(encoding="utf-8")


setup(
    name="page-entrokv",
    version="0.1.0",
    description=(
        "Entropy-guided, AST-aware KV-page selection for GQA paged attention "
        "serving engines"
    ),
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="Page-EntroKV contributors",
    license="Apache-2.0",
    url="https://github.com/bruce12-glitch/Entro-Prefill",
    packages=find_packages(
        include=["page_entrokv", "page_entrokv.*", "integration", "integration.*"]
    ),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "tree-sitter>=0.20,<0.22",
        "tree-sitter-languages>=1.10",
    ],
    extras_require={
        "hf": ["transformers>=4.36", "torch>=2.1", "accelerate"],
        "vllm": ["vllm>=0.4.0"],
        "dev": ["pytest>=7.0"],
        "all": ["transformers>=4.36", "torch>=2.1", "accelerate", "vllm>=0.4.0"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
