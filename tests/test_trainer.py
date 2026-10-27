import json
import pytest
from unittest.mock import MagicMock, patch
from datasets import Dataset
import training.trainer
from utils.file_utils import dataset_root
import utils.file_utils

# ---------------------------------------------------------------------------
# 1. Tests for load_and_format_dataset
# ---------------------------------------------------------------------------

def test_load_and_format_dataset_file_not_found(tmp_path, monkeypatch):
    """Ensure FileNotFoundError is raised when training file does not exist."""
    # Patch dataset_root where it is imported in trainer.py
    monkeypatch.setattr(training.trainer, "dataset_root", lambda: str(tmp_path))
    mock_tokenizer = MagicMock()

    with pytest.raises(FileNotFoundError, match="Dataset training file cannot be found"):
        training.trainer.load_and_format_dataset(mock_tokenizer, str(tmp_path))


def test_load_and_format_dataset_success(tmp_path, monkeypatch):
    """Verify JSONL rows are parsed and formatted using chat template."""

    training_dataset_file = "mock_training_dataset.jsonl"

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    file_path = data_dir / training_dataset_file

    sample_rows = [
        {"messages": [{"role": "user", "content": "How do I invest?"}, {"role": "assistant", "content": "Start with an index fund."}]},
        {"messages": [{"role": "user", "content": "What is an emergency fund?"}, {"role": "assistant", "content": "3-6 months of expenses."}]}
    ]
    with open(file_path, "w") as f:
        for row in sample_rows:
            f.write(json.dumps(row) + "\n")

    monkeypatch.setattr(training.trainer, "dataset_root", lambda: str(data_dir))

    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.side_effect = lambda msgs, tokenize=False: f"<formatted>{msgs[1]['content']}</formatted>"

    dataset = training.trainer.load_and_format_dataset(mock_tokenizer, training_dataset_file)

    assert isinstance(dataset, Dataset)
    assert len(dataset) == 2
    assert "text" in dataset.column_names
    assert dataset[0]["text"] == "<formatted>How do I invest?</formatted>"
    assert mock_tokenizer.apply_chat_template.call_count == 2


# ---------------------------------------------------------------------------
# 2. Tests for load_model_from_hf
# ---------------------------------------------------------------------------

@patch("training.trainer.torch.cuda.memory_allocated", return_value=4e9)
@patch("training.trainer.prepare_model_for_kbit_training")
@patch("training.trainer.AutoModelForCausalLM")
@patch("training.trainer.AutoTokenizer")
def test_load_model_from_hf_cuda(
    mock_tokenizer_cls,
    mock_model_cls,
    mock_kbit,
    mock_mem,
):
    """Test CUDA model loading with 4-bit quantization config."""
    mock_model = MagicMock()
    mock_model_cls.from_pretrained.return_value = mock_model
    mock_kbit.return_value = mock_model

    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = None
    mock_tokenizer.eos_token = "<eos>"
    mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

    model, tokenizer = training.trainer.load_model_from_hf("test-model", device_map="cuda", use_bf16=True)

    mock_model_cls.from_pretrained.assert_called_once()
    _, kwargs = mock_model_cls.from_pretrained.call_args
    assert kwargs["device_map"] == "cuda"
    assert kwargs["quantization_config"].load_in_4bit is True
    assert model == mock_model

    # Tokenizer assertions
    mock_tokenizer_cls.from_pretrained.assert_called_once_with("test-model", trust_remote_code=True)
    assert tokenizer.pad_token == "<eos>"

@patch("training.trainer.torch.cuda.memory_allocated", return_value=4e9)
@patch("training.trainer.prepare_model_for_kbit_training")
@patch("training.trainer.AutoModelForCausalLM")
@patch("training.trainer.AutoTokenizer")
def test_load_model_from_hf_cpu(
    mock_tokenizer_cls,
    mock_model_cls,
    mock_kbit,
    mock_mem,
):
    """Test CPU loading without quantization config."""
    mock_model = MagicMock()
    mock_model_cls.from_pretrained.return_value = mock_model
    mock_kbit.return_value = mock_model

    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = None
    mock_tokenizer.eos_token = "<eos>"
    mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

    model, tokenizer = training.trainer.load_model_from_hf("test-model", device_map="cpu", use_bf16=True)

    mock_model_cls.from_pretrained.assert_called_once()
    _, kwargs = mock_model_cls.from_pretrained.call_args
    assert kwargs["device_map"] == "cpu"
    assert "quantization_config" not in kwargs # no quantization for cpu mode
    assert model == mock_model

    # Tokenizer assertions
    mock_tokenizer_cls.from_pretrained.assert_called_once_with("test-model", trust_remote_code=True)
    assert tokenizer.pad_token == "<eos>"




def test_load_model_from_hf_invalid_device():
    """Verify that unsupported devices raise an Exception."""
    with pytest.raises(Exception, match="Unsupported device mps"):
        training.trainer.load_model_from_hf("test-model", device_map="mps", use_bf16=False)


# ---------------------------------------------------------------------------
# 3. Tests for train
# ---------------------------------------------------------------------------

@patch("training.trainer.output_root")
@patch("training.trainer.SFTTrainer")
@patch("training.trainer.SFTConfig")
def test_train_execution_and_artifact_saving(mock_sft_config, mock_sft_trainer_cls, mock_output_root, tmp_path):
    """Verify SFTTrainer execution and persistence of adapter + tokenizer."""
    mock_output_root.return_value = tmp_path
    mock_trainer_instance = MagicMock()
    mock_sft_trainer_cls.return_value = mock_trainer_instance

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_dataset = MagicMock()

    training.trainer.train(
        lr=2e-4,
        nb_epochs=1,
        training_model=mock_model,
        training_tokenizer=mock_tokenizer,
        training_dataset=mock_dataset,
        report_training="none",
        use_bf16=False,
        use_fp16=True,
    )

    # Check trainer train call and model/tokenizer saves
    mock_trainer_instance.train.assert_called_once()
    mock_trainer_instance.model.save_pretrained.assert_called_once_with(f"{tmp_path}/adapter")
    mock_tokenizer.save_pretrained.assert_called_once_with(f"{tmp_path}/tokenizer")


@patch("training.trainer.output_root")
@patch("training.trainer.SFTTrainer")
@patch("training.trainer.SFTConfig")
def test_train_fallback_signature_on_type_error(mock_sft_config, mock_sft_trainer_cls, mock_output_root, tmp_path):
    """Verify backward-compatibility fallback when processing_class causes TypeError."""
    mock_output_root.return_value = tmp_path

    # First call fails (TypeError on processing_class), second succeeds (tokenizer=...)
    mock_trainer_instance = MagicMock()
    mock_sft_trainer_cls.side_effect = [TypeError("unexpected keyword processing_class"), mock_trainer_instance]

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    training.trainer.train(
        lr=1e-4,
        nb_epochs=1,
        training_model=mock_model,
        training_tokenizer=mock_tokenizer,
        training_dataset=MagicMock(),
        report_training="none",
        use_bf16=False,
        use_fp16=False,
    )

    assert mock_sft_trainer_cls.call_count == 2
    mock_trainer_instance.train.assert_called_once()


# ---------------------------------------------------------------------------
# 4. Tests for run_training (Orchestration)
# ---------------------------------------------------------------------------

@patch("training.trainer.train")
@patch("training.trainer.load_and_format_dataset")
@patch("training.trainer.load_model_from_hf")
@patch("training.trainer.torch.cuda.is_bf16_supported", return_value=True)
@patch("training.trainer.get_device", return_value="cuda")
def test_run_training_pipeline(mock_get_device, mock_bf16, mock_load_model, mock_load_data, mock_train):
    """Test full training pipeline integration with mocked subroutines."""
    training_dataset_file = "mock_training_dataset.jsonl"

    mock_settings = MagicMock()
    mock_settings.model_name = "mistral-7b"
    mock_settings.learning_rate = 1e-4
    mock_settings.nb_epochs = 3
    mock_settings.report_training = "none"
    mock_settings.training_set = training_dataset_file

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_load_model.return_value = (mock_model, mock_tokenizer)
    mock_dataset = MagicMock()
    mock_load_data.return_value = mock_dataset

    training.trainer.run_training(mock_settings)

    mock_load_model.assert_called_once_with(model_name="mistral-7b", device_map="cuda", use_bf16=True)
    mock_load_data.assert_called_once_with(mock_tokenizer, training_dataset_file)
    mock_train.assert_called_once_with(
        lr=1e-4,
        nb_epochs=3,
        training_model=mock_model,
        training_tokenizer=mock_tokenizer,
        training_dataset=mock_dataset,
        report_training="none",
        use_bf16=True,
        use_fp16=False,
    )