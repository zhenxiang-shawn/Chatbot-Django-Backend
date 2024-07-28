"""
Entity extraction class using a BERT model for token classification.

This module defines the EntityExtractor class, which uses a pre-trained BERT model to
classify tokens and extract entities from input text.
"""

import json
import torch
from transformers import BertForTokenClassification, BertTokenizer


class EntityExtractor:
    """Class for extracting entities from text using a BERT model."""

    def __init__(self, saved_model_dir, max_length=128):
        """
        Initialize the EntityExtractor with a pre-trained BERT model.

        Args:
            saved_model_dir (str): The directory where the saved model is located.
            max_length (int, optional): The maximum length of the input text. Defaults to 128.
        """
        try:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertForTokenClassification.from_pretrained(saved_model_dir)
            tag_filepath = "entityExtractor/data/tag2id.json"
            with open(tag_filepath, 'r', encoding='utf-8') as file:
                self.tag2id = json.load(file)
        except Exception as e:
            raise IOError(f"Error loading the model: {e}")

        self.max_length = max_length
        self.id2tag = {v: k for k, v in self.tag2id.items()}

    def predict_and_decode(self, text):
        """
        Predict and decode entities in the input text.

        Args:
            text (str): The input text to process.

        Returns:
            list of tuples: A list of (entity, tag) tuples.
        """
        # Encode the text
        encoded_input = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoded_input['input_ids']
        attention_mask = encoded_input['attention_mask']

        # Model prediction
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask=attention_mask).logits
        predictions = torch.argmax(logits, dim=-1)

        # Decode prediction results
        decoded_output = [
            (self.id2tag[pred.item()], token)
            for pred, token in zip(predictions[0], self.tokenizer.convert_ids_to_tokens(input_ids[0]))
        ]

        return self.extract_entities(decoded_output)

    def extract_entities(self, decoded_output):
        """
        Extract entities from the decoded output.

        Args:
            decoded_output (list of tuples): The decoded output from the model.

        Returns:
            dict: A dictionary containing extracted entities.
        """
        # Initialize entity buffers
        m_entities = {
            'command': None,
            'quantity': None,
            'attributes': [],
            'object': None
        }
        attribute_buffer, object_buffer, quantity_buffer = [], [], []

        for tag, token in decoded_output:
            if tag == 'B-CMD':
                m_entities['command'] = token
            elif tag == 'I-CMD':
                m_entities['command'] += ' ' + token
            # ... (rest of the method remains the same)

        # Process the buffers to set the final entities
        # ...

        return m_entities


"""
Test code for entity extraction class.
"""
# ee = EntityExtractor('./my_bert_ner_model/')
# user_input = "can you delete the soft white cube"
# entities = ee.predict_and_decode(user_input)
# print(entities)
