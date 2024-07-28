## Chatbot Backend with NER Model

This project is a Django backend built with chatbot and NER model. The chatbot was built by using [chatterbot](https://chatterbot.readthedocs.io/en/stable/). The NER model used [BERT](https://huggingface.co/docs/transformers/model_doc/bert) as a pretrained model to extract command, such as "remove", "create", "delete", and related attributes.

Author: Shawn Jin (zhenxiang.shawn@zohomail.com)

This backend has two response options:
1. return the chatbot response and entities extracted by NER model all the time.
2. return the chatbot response when the command is not find or not implemented yet, or else, return positive reply to confirm the request comes from frontend.

 This option could be modified in `djangoProject/controller/chat.py`

> *Note*
>
> The frontend need to set up CSRF verification to send POST request to `api/chat`
> 
> SECRET_KEY = "django-insecure-+egsx_2(^(hc=92q#))b*st9u5tn&w=x3)k%gc56#mgze0x^ll" This is ONLY FOR TEST

#### Routers

- `[GET] admin/`: The main page of this project. (TODO:zhenxiang Need to finish the login & sign up logic)
- `[POST] api/chat`: The chat api. Need add a request body in `application/json` format. For example: 
  ``` json
  {"query": "how are you"}
  ```

### Set up 

### install Packages

1. **Only support Python 3.8**.
2. update pypi before installing packages.
`pip install --upgrade pip`

3. install chatterbot `pip install pytz blis chatterbot`
4. install pytorch & transformer `pip install transformers torch`

### Run server

`python manage.py runserver`

