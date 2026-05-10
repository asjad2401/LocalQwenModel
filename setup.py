from setuptools import setup, find_packages

setup(
    name="local-agent",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "click>=8.1.0",
        "rich>=13.7.0",
        "fastapi>=0.115.0",
        "uvicorn[standard]>=0.32.0",
        "httpx>=0.27.0",
        "chromadb>=0.5.23",
        "anthropic>=0.40.0",
        "openai>=1.50.0",
        "pyyaml>=6.0.2",
        "gitpython>=3.1.43",
        "python-multipart>=0.0.12",
    ],
    entry_points={
        "console_scripts": [
            "agent=agent.cli:cli",
        ],
    },
    python_requires=">=3.12",
)
