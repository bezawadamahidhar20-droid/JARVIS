"""Test the Ollama client: availability check + one generation."""

import config
from ai.ollama import OllamaClient
from utils import logger


def main():
    client = OllamaClient()
    client.check_available()

    question = "What is the capital of Japan? Answer in one short sentence."
    logger.status(f"Q: {question}")

    start = logger.tick()
    answer = client.generate(question)
    logger.report("OLLAMA", start)

    print(f"\nJARVIS: {answer}")


if __name__ == "__main__":
    main()