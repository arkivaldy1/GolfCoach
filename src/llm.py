import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_anthropic import ChatAnthropic

load_dotenv()

def get_llm(provider: str = "Ollama", model: str | None = None):
    """
    Return a configured chat model

    Switch providers by changing one argument. Business logic stays the same.
    """

    if provider == "ollama":
        return ChatOllama(
            model=model or "llama3.2",
            temperature=0.3,
        )
    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set.")

        return ChatAnthropic(
            model=model or "claude-haiku-4-5",
            temperature=0.3,
            api_key=api_key,
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}. Supported providers are 'ollama' and 'anthropic'.")