import os.path

from django.apps import AppConfig
from chatbot.model import ChatterBotInstance


# In the ready method of ChatbotAppConfig, an instance of ChatterBot is initialized
class ChatbotAppConfig(AppConfig):
    """
    This class represents the configuration for the 'chat_app_backend' application.

    The __init__ method initializes the base AppConfig and sets the initial state.
    """
    name = 'chatbot'

    def __init__(self, app_name, app_module):
        """
        Initialize the ChatbotAppConfig instance.

        Args:
            app_name (str): The name of the application.
            app_module (module): The module of the application.
        """
        super().__init__(app_name, app_module)
        self.chatbot_instance = None

    def ready(self):
        """
        Perform actions when the application is ready.

        This method creates a ChatterBot instance and handles training if the database doesn't exist.
        """
        # Create a ChatterBot instance
        # Make the chatbot_instance available at the application level by adding it to the AppConfig
        self.chatbot_instance = ChatterBotInstance(
            name='MyChatterBot',
            storage_adapter='chatterbot.storage.SQLStorageAdapter',
            database_uri='sqlite:///database.sqlite3'
        )
        if not os.path.exists('database.sqlite3'):
            """
            Train the ChatterBot instance if the database doesn't exist.
            """
            self.chatbot_instance.train()