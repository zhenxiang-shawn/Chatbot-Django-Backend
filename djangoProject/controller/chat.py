from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
import json
from http import HTTPStatus
from chatbot.model import ChatterBotInstance
from entityExtractor.model import EntityExtractor

# Constants
SAVED_MODEL_PATH = "entityExtractor/my_bert_ner_model/"
CHATBOT = ChatterBotInstance(
    name="ChatBot",
    storage_adapter="chatterbot.storage.SQLStorageAdapter",
    database_uri='sqlite:///database.sqlite3'
)


@require_http_methods(["POST"])
def chat_view(request, compose_response=False):
    """
    Handle the chat view logic, which includes processing POST requests,
    extracting entities, generating responses, and returning JSON responses.
    """
    try:
        # Load query from the request body
        query = json.loads(request.body.decode('utf-8')).get("query", None)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON in request'}, status=HTTPStatus.BAD_REQUEST)

    if not query:
        return JsonResponse({'error': 'No query provided'}, status=HTTPStatus.BAD_REQUEST)

    # Use EntityExtractor to predict and decode entities from the query
    entities = EntityExtractor(SAVED_MODEL_PATH).predict_and_decode(query)

    # Get response from the chatbot
    response = CHATBOT.query(query)

    # If compose_response is True, return a JsonResponse with additional data
    if compose_response:
        return JsonResponse(
            data={
                'message': 'Chat API works!',
                'entities': str(entities),
                'response': str(response)
            },
            status=HTTPStatus.OK
        )

    # Load the command pool from a JSON file
    with open("chatbot/command_pool.json", "r", encoding='utf-8') as file:
        command_pool = json.load(file)

    # Check if the extracted command is in the command pool
    command = entities.get("command", None)
    if command and command in command_pool['commands']:
        response = f"Processing {command}..."

    # Return a JsonResponse with the message, entities, and response
    return JsonResponse(
        data={
            'message': 'Chat API works!',
            'entities': str(entities),
            'response': str(response)
        },
        status=HTTPStatus.OK
    )
