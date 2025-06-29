from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="generativelife",
    version="1.0.0",
    description="LLM-Powered Agent Simulation - Advanced artificial life simulation with AI agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="GenerativeLife Contributors",
    url="https://github.com/dexmac221/GenerativeLife",
    packages=find_packages(),
    install_requires=[
        "pygame>=2.5.2",
        "matplotlib>=3.8.2",
        "numpy>=1.26.2",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
        "ollama>=0.2.1",
        "pyyaml>=6.0.1",
        "openai>=1.0.0",
    ],
    python_requires=">=3.11",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Games/Entertainment :: Simulation",
    ],
    entry_points={
        "console_scripts": [
            "generativelife=ai_arena:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "prompts/*"],
    },
)
