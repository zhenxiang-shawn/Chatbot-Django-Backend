from django.http import HttpResponse


class TokenCheckMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
        #######################################################################
        # WARNING: In this example, a hard-coded token is used for illustration
        # purposes. NEVER use a fixed token in a production environment.
        # Tokens should be dynamically generated and validated through secure
        # means such as OAuth2.0.
        self.TOKEN = "Bearer 6dc415d4-7872-41d8-bd80-8ffe3ef4f4e8"
        #######################################################################

    def __call__(self, request):
        # Check if the request method is POST and if the 'token' is present in
        # the POST parameters
        if request.method == 'POST' and 'token' in request.POST:
            if request.POST.get('token') == self.TOKEN:
                return self.get_response(request)
            else:
                return HttpResponse('Invalid token', status=401)
        elif request.method == 'POST':
            return HttpResponse('Token not provided', status=400)

        return self.get_response(request)
