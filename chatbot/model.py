import ssl
import nltk
from chatterbot import ChatBot

from chatterbot.trainers import ChatterBotCorpusTrainer
from chatterbot.trainers import ListTrainer

from chatbot.train_data import custom_training_data_list

# Set ssl context
ssl._create_default_https_context = ssl._create_unverified_context

# Download punkt for tokenizer
nltk.download('punkt')


class ChatterBotInstance(ChatBot):
    """
    This class represents an instance of a ChatterBot with custom configuration and training methods.

    Attributes:
        name (str): The name of the chatbot.
        storage_adapter (str): The storage adapter used for the chatbot.
        database_uri (str): The URI of the database for the chatbot.
    """

    def __init__(self, name: str, storage_adapter: str, database_uri: str, **kwargs):
        """
        Initialize the ChatterBotInstance.

        Args:
            name (str): The name of the chatbot.
            storage_adapter (str): The storage adapter to be used.
            database_uri (str): The URI of the database.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(name, **kwargs)

        self.name = name
        self.storage_adapter = storage_adapter
        self.database_uri = database_uri
        self._setup_chatbot()

    def _setup_chatbot(self):
        """
        Configure the internal ChatBot instance with the given settings.
        """
        self.chatbot = ChatBot(name=self.name,
                               storage_adapter=self.storage_adapter,
                               database_uri=self.database_uri)

    def query(self, sentence: str) -> str:
        """
        Query the chatbot with the provided sentence and return the response.

        Args:
            sentence (str): The input sentence.

        Returns:
            str: The response from the chatbot.
        """
        return self.chatbot.get_response(sentence)

    def train(self):
        """
        Train the chatbot using the default corpus and custom training data.
        """
        # Create trainner instance
        trainer = ChatterBotCorpusTrainer(self.chatbot)

        # Train the chatbot by using default corpus
        trainer.train("chatterbot.corpus.english",
                      'chatterbot.corpus.english.greetings')

        # Train chatterbot with custom training data
        list_trainer = ListTrainer(self.chatbot)
        for custom_training_data in custom_training_data_list:
            list_trainer.train(custom_training_data)

        # Save after training. Do NOT need to use saving function, due to 'SQLStorageAdapter' will save automatically.
        # chatbot.storage.save()
