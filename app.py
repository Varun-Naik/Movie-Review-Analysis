from flask import Flask
import transformers
from transformers import TFBertForSequenceClassification
from transformers import BertTokenizer


# print a nice greeting.
def say_hello(username="World"):
    return '<p>Hello %s!</p>\n' % username


# some bits of text for the page.
header_text = '''
    <html>\n<head> <title>EB Flask Test</title> </head>\n<body>'''
instructions = '''
    <p><em>Hint</em>: This is a RESTful web service! Append a username
    to the URL (for example: <code>/Thelonious</code>) to say hello to
    someone specific.</p>\n'''
home_link = '<p><a href="/">Back</a></p>\n'
footer_text = '</body>\n</html>'


# add a rule for the index page.
app.add_url_rule('/', 'index', (lambda: header_text +
                                        say_hello() + instructions + footer_text))

# add a rule when the page is accessed with a name appended to the site
# URL.
app.add_url_rule('/<username>', 'hello', (lambda username:
                                          header_text + say_hello(username) + home_link + footer_text))

# Load the fine-tuned BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
best_model = TFBertForSequenceClassification.from_pretrained("E:\Programming files\flaskmovie\imdb_bert_pretrained")


# Prediction function
def get_prediction(message, model):
    # inference
    results = model(message)
    return results


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


if __name__ == '__main__':
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    app.debug = True
    app.run()
