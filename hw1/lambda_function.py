"""
In this file we specify default event handlers which are then populated into the handler map using metaprogramming
Copyright Anjishnu Kumar 2015
Happy Hacking!
"""

from ask import alexa

def lambda_handler(request_obj, context=None):
    '''
    This is the main function to enter to enter into this code.
    If you are hosting this code on AWS Lambda, this should be the entry point.
    Otherwise your server can hit this code as long as you remember that the
    input 'request_obj' is JSON request converted into a nested python object.
    '''

    metadata = {'h' : 'SomeRandomDude'} # add your own metadata to the request using key value pairs
    
    ''' inject user relevant metadata into the request if you want to, here.    
    e.g. Something like : 
    ... metadata = {'user_name' : some_database.query_user_name(request.get_user_id())}

    Then in the handler function you can do something like -
    ... return alexa.create_response('Hello there {}!'.format(request.metadata['user_name']))
    '''
    return alexa.route_request(request_obj, metadata)


@alexa.default_handler()
def default_handler(request):

    """ The default handler gets invoked if no handler is set for a request """
    return alexa.create_response(message=request.get_slot_map()["Text"])


@alexa.request_handler("LaunchRequest")
def launch_request_handler(request):
    return alexa.create_response(message="Hello Welcome to this basic Alexa demo!")


@alexa.request_handler("SessionEndedRequest")
def session_ended_request_handler(request):
    return alexa.create_response(message="Goodbye!")

@alexa.intent_handler('GetRecipeIntent')
def get_recipe_intent_handler(request):
    """
    You can insert arbitrary business logic code here    
    """

    # Get variables like userId, slots, intent name etc from the 'Request' object
    ingredient = request.slots["Ingredient"] 

    if ingredient == None:
        return alexa.create_response("Could not find an ingredient!")

    card = alexa.create_card(title="GetRecipeIntent activated", subtitle=None,
                             content="asked alexa to find a recipe using {}".format(ingredient))
    
    return alexa.create_response("Finding a recipe with the ingredient {}".format(ingredient),
                                 end_session=False, card_obj=card)



@alexa.intent_handler('NextRecipeIntent')
def next_recipe_intent_handler(request):
    """
    You can insert arbitrary business logic code here
    """
    return alexa.create_response(message="Getting Next Recipe ... 123")
