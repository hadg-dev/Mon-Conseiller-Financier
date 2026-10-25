from settings import Settings
from validation.judge import score_with_llm
from validation.perplexity import score_perplexity

def run_validation(settings: Settings) -> None:
    #score_perplexity(settings)
    score_with_llm(settings)
