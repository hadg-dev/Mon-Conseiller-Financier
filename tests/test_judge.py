import json
import logging
from unittest.mock import MagicMock, mock_open, patch
import pytest
import torch

from validation.judge import JUDGE_MODEL, JUDGE_PROMPT, score_with_llm


@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.model_name = "test-model-name"
    settings.validation_set = "eval_data.jsonl"
    return settings


@pytest.fixture
def sample_dataset_lines():
    sample_records = [
        {
            "messages": [
                {"role": "user", "content": "What is IRR?"},
                {"role": "assistant", "content": "Internal Rate of Return."},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Explain NPV."},
                {"role": "assistant", "content": "Net Present Value."},
            ]
        },
    ]
    return "\n".join(json.dumps(r) for r in sample_records)


def test_score_with_llm_missing_api_key_skips(mock_settings, monkeypatch, caplog):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with caplog.at_level(logging.WARNING), patch("validation.judge.OpenAI") as mock_openai:
        score_with_llm(mock_settings)

        assert "no OPENAI_API_KEY configured in env; skipping" in caplog.text
        mock_openai.assert_not_called()


@patch("validation.judge.get_device", return_value="cpu")
@patch("validation.judge.output_root", return_value="/fake/output")
@patch("validation.judge.dataset_root", return_value="/fake/dataset")
@patch("validation.judge.AutoTokenizer.from_pretrained")
@patch("validation.judge.AutoModelForCausalLM.from_pretrained")
@patch("validation.judge.PeftModel.from_pretrained")
@patch("validation.judge.OpenAI")
def test_score_with_llm_full_flow(
    mock_openai_cls,
    mock_peft_from_pretrained,
    mock_automodel_from_pretrained,
    mock_tokenizer_from_pretrained,
    mock_dataset_root,
    mock_output_root,
    mock_get_device,
    mock_settings,
    sample_dataset_lines,
    monkeypatch,
    caplog,
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    # Mock Tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.eos_token_id = 50256
    mock_tokenized_input = MagicMock()
    mock_tokenized_input.input_ids.shape = (1, 4)
    mock_tokenized_input.to.return_value = mock_tokenized_input
    mock_tokenizer.return_value = mock_tokenized_input
    mock_tokenizer.decode.return_value = "Model generated answer."
    mock_tokenizer_from_pretrained.return_value = mock_tokenizer

    # Mock Model & PEFT
    mock_base_model = MagicMock()
    mock_automodel_from_pretrained.return_value = mock_base_model

    mock_peft_model = MagicMock()
    # Mock tensor output for model.generate
    mock_peft_model.generate.return_value = torch.tensor([[10, 11, 12, 13, 14, 15]])
    mock_peft_from_pretrained.return_value = mock_peft_model

    # Mock OpenAI client response
    mock_client = MagicMock()
    mock_completion_1 = MagicMock()
    mock_completion_1.choices = [
        MagicMock(message=MagicMock(content="Score: 4\nJustification: Good answer."))
    ]
    mock_completion_2 = MagicMock()
    mock_completion_2.choices = [
        MagicMock(message=MagicMock(content="Score: 5\nJustification: Perfect answer."))
    ]
    mock_client.chat.completions.create.side_effect = [
        mock_completion_1,
        mock_completion_2,
    ]
    mock_openai_cls.return_value = mock_client

    with caplog.at_level(logging.INFO), patch(
        "builtins.open", mock_open(read_data=sample_dataset_lines)
    ):
        score_with_llm(mock_settings)

    # Verifications
    mock_get_device.assert_called_once()
    mock_tokenizer_from_pretrained.assert_called_once_with(
        mock_settings.model_name, trust_remote_code=True
    )
    mock_automodel_from_pretrained.assert_called_once_with(
        mock_settings.model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    mock_peft_from_pretrained.assert_called_once_with(
        mock_base_model, "/fake/output/adapter"
    )
    mock_peft_model.eval.assert_called_once()

    assert mock_peft_model.generate.call_count == 2
    assert mock_client.chat.completions.create.call_count == 2

    first_call_kwargs = mock_client.chat.completions.create.call_args_list[0].kwargs
    assert first_call_kwargs["model"] == JUDGE_MODEL
    assert first_call_kwargs["temperature"] == 0.0
    assert first_call_kwargs["messages"][0] == {
        "role": "system",
        "content": JUDGE_PROMPT,
    }
    assert "What is IRR?" in first_call_kwargs["messages"][1]["content"]
    assert "Internal Rate of Return." in first_call_kwargs["messages"][1]["content"]

    assert "Number of scored samples : 2/2" in caplog.text
    assert "Average score : 4.50 / 5" in caplog.text


@patch("validation.judge.get_device", return_value="cpu")
@patch("validation.judge.output_root", return_value="/fake/output")
@patch("validation.judge.dataset_root", return_value="/fake/dataset")
@patch("validation.judge.AutoTokenizer.from_pretrained")
@patch("validation.judge.AutoModelForCausalLM.from_pretrained")
@patch("validation.judge.PeftModel.from_pretrained")
@patch("validation.judge.OpenAI")
def test_score_with_llm_handles_non_matching_judge_output(
    mock_openai_cls,
    mock_peft_from_pretrained,
    mock_automodel_from_pretrained,
    mock_tokenizer_from_pretrained,
    mock_dataset_root,
    mock_output_root,
    mock_get_device,
    mock_settings,
    monkeypatch,
    caplog,
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    mock_tokenizer = MagicMock()
    mock_tokenized_input = MagicMock()
    mock_tokenized_input.input_ids.shape = (1, 2)
    mock_tokenized_input.to.return_value = mock_tokenized_input
    mock_tokenizer.return_value = mock_tokenized_input
    mock_tokenizer_from_pretrained.return_value = mock_tokenizer

    mock_peft_model = MagicMock()
    mock_peft_model.generate.return_value = torch.tensor([[1, 2, 3]])
    mock_peft_from_pretrained.return_value = mock_peft_model

    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_completion.choices = [
        MagicMock(message=MagicMock(content="Invalid response format without score tag"))
    ]
    mock_client.chat.completions.create.return_value = mock_completion
    mock_openai_cls.return_value = mock_client

    single_line_data = json.dumps(
        {
            "messages": [
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Answer"},
            ]
        }
    )

    with caplog.at_level(logging.INFO), patch(
        "builtins.open", mock_open(read_data=single_line_data)
    ):
        score_with_llm(mock_settings)

    assert "Number of scored samples : 0/1" in caplog.text
    assert "Average score : 0.00 / 5" in caplog.text