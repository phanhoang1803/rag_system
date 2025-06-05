# intelligent_rag_system/src/query_understanding/ner_extractor.py

import spacy
from typing import List, Dict, Any

class NERExtractor:
    """
    Extracts Named Entities (NER) from text using SpaCy.
    Focuses solely on NER extraction.
    """
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initializes the SpaCy NER model.
        Downloads the model if not already present.
        """
        try:
            self.nlp = spacy.load(model_name)
            print(f"SpaCy NER model '{model_name}' loaded successfully.")
        except OSError:
            print(f"SpaCy model '{model_name}' not found. Downloading...")
            spacy.cli.download(model_name)
            self.nlp = spacy.load(model_name)
            print(f"SpaCy NER model '{model_name}' downloaded and loaded.")
        except Exception as e:
            print(f"Error loading SpaCy model: {e}")
            raise
    
    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """
        Extracts named entities from the given text.

        Args:
            text: The input string (e.g., user query).

        Returns:
            A list of dictionaries, each containing 'text', 'label', and 'start_char', 'end_char'
            for the extracted entities.
        """
        if not text:
            return []
        
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char
            })
        return entities
    
# Example Usage
if __name__ == "__main__":
    ner_extractor = NERExtractor()

    query1 = "What is the price of Cloud Storage Premium?"
    entities1 = ner_extractor.extract_entities(query1)
    print(f"Query: '{query1}'\nEntities: {entities1}\n")

    query2 = "Who is Alice Smith in the HR department?"
    entities2 = ner_extractor.extract_entities(query2)
    print(f"Query: '{query2}'\nEntities: {entities2}\n")

    query3 = "Tell me about the latest Q2 report."
    entities3 = ner_extractor.extract_entities(query3)
    print(f"Query: '{query3}'\nEntities: {entities3}\n")

    query4 = "How does Product A compare to Product B?"
    entities4 = ner_extractor.extract_entities(query4)
    print(f"Query: '{query4}'\nEntities: {entities4}\n")