from unittest.mock import MagicMock
import pytest
from src.prepare_dataset import calculate_token_count

def test_calculate_token_count_list():
    # Case 1: apply_chat_template with return_dict=False successfully returns a list of tokens
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = [1, 2, 3, 4, 5]
    
    messages = [{"role": "user", "content": "hello"}]
    count = calculate_token_count(messages, mock_tokenizer)
    
    assert count == 5
    mock_tokenizer.apply_chat_template.assert_called_once_with(messages, tokenize=True, return_dict=False)

def test_calculate_token_count_dict():
    # Case 2: apply_chat_template with return_dict=False raises an error, 
    # but apply_chat_template without return_dict returns a dict containing input_ids
    mock_tokenizer = MagicMock()
    
    def side_effect(*args, **kwargs):
        if "return_dict" in kwargs:
            raise TypeError("Unexpected keyword argument 'return_dict'")
        return {"input_ids": [10, 20, 30], "attention_mask": [1, 1, 1]}
        
    mock_tokenizer.apply_chat_template.side_effect = side_effect
    
    messages = [{"role": "user", "content": "hello"}]
    count = calculate_token_count(messages, mock_tokenizer)
    
    assert count == 3

def test_calculate_token_count_fallback_encode():
    # Case 3: apply_chat_template fails completely, fallback to encode
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.side_effect = RuntimeError("Template error")
    mock_tokenizer.encode.return_value = [1, 2, 3, 4]
    
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"}
    ]
    count = calculate_token_count(messages, mock_tokenizer)
    
    assert count == 4
    mock_tokenizer.encode.assert_called_once_with("sys\nusr")
