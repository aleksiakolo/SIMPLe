import torch
from transformers import AutoTokenizer, MBartForConditionalGeneration
from loguru import logger

def test_translation_model(model, tokenizer, sample_text, max_length=128):
    """Test function to check if the model's decoding outputs are valid."""
    model.eval()  # Set the model to evaluation mode
    
    with torch.no_grad():
        # Tokenize the input
        inputs = tokenizer(sample_text, return_tensors="pt", max_length=max_length, truncation=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # Generate outputs
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length)
        
        # Decode the outputs
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Log the results
        if decoded_outputs and all(decoded_outputs):
            logger.info(f"Decoded Outputs: {decoded_outputs}")
        else:
            logger.warning("Decoded outputs are None or empty. Check the model or input data.")
        return decoded_outputs

# Example usage for testing
if __name__ == "__main__":
    # Initialize the tokenizer and model
    model_name = "facebook/mbart-large-50"  # or "t5-small" for T5Translation
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(model_name)  # Replace with your initialized model if needed
    
    # Sample input text to test the model
    sample_text = "This is a test sentence for translation."
    
    # Run the test
    decoded_result = test_translation_model(model, tokenizer, sample_text)
