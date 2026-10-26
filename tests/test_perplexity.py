
import json
import math
from unittest.mock import MagicMock, mock_open, patch
import pytest
import torch
from validation.perplexity import score_perplexity

@pytest.fixture
def dummy_settings():
    """Provides mock settings."""
    settings = MagicMock()
    settings.model_name = "dummy-org/dummy-model"
    settings.validation_set = "eval.jsonl"
    return settings


@pytest.fixture
def mock_dataset_content():
    """Sample JSONL records simulating chat messages."""
    records = [
        {
            "messages": [
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "It is 4."},
            ]
        },
    ]
    return "\n".join(json.dumps(r) for r in records)


@pytest.fixture
def mock_dependencies():
    """Mocks all expensive or I/O-bound components."""
    with (
        patch("validation.perplexity.get_device") as mock_device,
        patch("validation.perplexity.output_root", return_value="/tmp/output"),
        patch("validation.perplexity.dataset_root", return_value="/tmp/dataset"),
        patch("validation.perplexity.AutoTokenizer.from_pretrained") as mock_tok_cls,
        patch("validation.perplexity.AutoModelForCausalLM") as mock_base_model_cls,
        patch("validation.perplexity.PeftModel.from_pretrained") as mock_peft_cls,
    ):
        # 1. Device
        mock_device.return_value = "cpu"

        # 2. Tokenizer behavior
        mock_tokenizer = MagicMock()
        # Mock apply_chat_template to return dummy 2D tensors
        def fake_apply_chat_template(messages, add_generation_prompt=False, return_tensors="pt"):
            length = 4 if add_generation_prompt else 8
            # Return a dict containing "input_ids"
            return {"input_ids": torch.zeros((1, length), dtype=torch.long)}

        mock_tokenizer.apply_chat_template.side_effect = fake_apply_chat_template
        mock_tok_cls.return_value = mock_tokenizer

        # 3. Model forward pass
        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.loss = torch.tensor(1.5)  # Fixed mock loss
        mock_model.return_value = mock_output
        mock_base_model_cls.from_pretrained.return_value = mock_model
        mock_peft_cls.return_value = mock_model

        yield {
            "device": mock_device,
            "tokenizer": mock_tokenizer,
            "model": mock_model,
            "peft_cls": mock_peft_cls,
            "base_model_cls": mock_base_model_cls,
        }


def test_score_perplexity_success(dummy_settings, mock_dependencies, mock_dataset_content, caplog):
    """Tests the full flow when valid records are evaluated."""
    caplog.set_level("INFO")

    with patch("builtins.open", mock_open(read_data=mock_dataset_content)):
        score_perplexity(dummy_settings)

    # Verify model initialization
    mock_dependencies["base_model_cls"].from_pretrained.assert_called_once_with(
        dummy_settings.model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    # FIXME
    #mock_dependencies["peft_cls"].assert_called_once_with(
    #    mock_dependencies["base_model_cls"].return_value,
    #    "/tmp/output/adapter",
    #)
    mock_dependencies["model"].eval.assert_called_once()

    # Verify calculations in logs:
    # 2 records, each has 8 tokens - 4 prompt tokens = 4 target tokens.
    # Total tokens = 8. Loss = 1.5. Expected PPL = exp(1.5) ~= 4.48
    expected_ppl = math.exp(1.5)
    assert f"evaluated tokens : 8" in caplog.text
    assert f"negative mean Log-Likelihood : 1.5000" in caplog.text
    assert f"perplexity (PPL) : {expected_ppl:.2f}" in caplog.text


def test_score_perplexity_empty_dataset(dummy_settings, mock_dependencies, caplog):
    """Tests that function gracefully handles an empty dataset without crashing."""
    with patch("builtins.open", mock_open(read_data="")):
        score_perplexity(dummy_settings)

    assert "Aucun token cible évalué." in caplog.text


def test_score_perplexity_dict_tokenizer_output(dummy_settings, mock_dependencies, mock_dataset_content):
    """Verifies handling when apply_chat_template returns a dict/BatchEncoding."""
    # Override tokenizer to return a dict containing input_ids
    mock_dependencies["tokenizer"].apply_chat_template.side_effect = lambda msgs, add_generation_prompt=False, return_tensors="pt": {
        "input_ids": torch.zeros((1, 4 if add_generation_prompt else 8), dtype=torch.long)
    }

    with patch("builtins.open", mock_open(read_data=mock_dataset_content)):
        # Should execute without TypeError
        score_perplexity(dummy_settings)

    assert mock_dependencies["model"].call_count == 2