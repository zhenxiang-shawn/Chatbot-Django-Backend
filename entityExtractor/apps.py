"""
Django application configuration for the EntityExtractor app.

This module provides the configuration class for the EntityExtractor Django app.
"""

from django.apps import AppConfig

# Import the EntityExtractor model.
from entityExtractor.model import EntityExtractor


class EntityExtractorAppConfig(AppConfig):
    """
    Django AppConfig subclass for the EntityExtractor app.
    """

    # Set the name of the application.
    name = "entityExtractor"

    def __init__(self, app_name, app_module):
        """
        Initialize the EntityExtractorAppConfig instance.

        Args:
            app_name (str): The name of the application.
            app_module (module): The module of the application.
        """
        print(f"EntityExtractorAppConfig: appname: {app_name}, module: {app_module}")
        super().__init__(app_name, app_module)
        self.entity_extractor = None

    def ready(self):
        """
        Perform initialization steps when the app is ready.
        """
        self.entity_extractor = EntityExtractor("entityExtractor/my_bert_ner_model")