from flask import Flask, jsonify, render_template, request, session
from flask_session import Session

import os
from datetime import datetime
from flask_socketio import SocketIO, emit
import classification, config
import openai
import json
from dotenv import load_dotenv

# DREW HELPER FUNCTIONS
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from PyPDF2 import PdfReader
import io
import base64
# get the API key from the .env file
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = API_KEY
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY','poop')
app.config["SECRET_KEY"] = 'poop'

Session(app)

socketio = SocketIO(app, async_mode='gevent')

conversation_log = []
# socketio = SocketIO(app, cors_allowed_origins="*")

for i in range(10):
    socketio.emit('response', {"message": i})  # send the chunk message and result_dict to the client


@app.route('/')
@app.route('/home')
def home():
    # set secret key
    app.config["SECRET_KEY"] = 'poop'
    session["SECRET_KEY"] = 'poop'
    session['conversation'] = None
    session['chat_history'] = None

    """Renders the home page."""
    return render_template(
        'index.html',
        title='Home Page',
        year=datetime.now().year,
    )

@socketio.on('message')
def handle_message(data):
    full_message = ''
    load_dotenv()
    API_KEY = os.getenv("OPENAI_API_KEY")
    print(API_KEY)
    openai.api_key = API_KEY  # ensure API key is set locally
    conversation_log = session.get('conversation_log', [])
    
    if len(conversation_log) < 2:
        prompt = "You are Machinelle, a store clerk for Direct Machines, a metal machinery ecommerce retailer. Please handle the following customer request, being sure to be as helpful as possible:\n"+ data['prompt']
    else:
        prompt = data['prompt']
    temperature = .8
    result_dict = classification.query_intent(data['prompt'])
    intent, entities, query, results = result_dict
    # NOW RUN THE PROMPT:
    print("Result dict: ", result_dict)
    # Update the conversation log before responding
    conversation_log.append({"role": "user", "content": prompt})
    if len(conversation_log) > 5:
        conversation_log.pop(0)  # remove the oldest message if more than 5 messages

    # socketio.emit('response', {"message":"", "intent": intent, "entities": [(f" {entity}: {entities[entity]}") for entity in entities], "query": query, "results": results}) 
    print(conversation_log)
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=conversation_log,
        temperature=temperature,
        stream=True,  # set stream=True
        max_tokens=250
    )
    for chunk in response:
        chunk_message = chunk['choices'][0]['delta']  # extract the message
        print("CHUNKY",chunk_message)
        socketio.emit('response', {"message": chunk_message})  # send the chunk message and result_dict to the client
        try:
            full_message += chunk_message["content"]
        except:
            continue # if the message is not a string, skip it
    if len(conversation_log) > 5:
        conversation_log.pop(0)  # remove the oldest message if more than 5 messages
    conversation_log.append({"role": "assistant", "content": full_message})

    session['conversation_log'] = conversation_log
   
# ENDPOINT FOR CHATGPT API
@app.route('/api/ChatGPTWebAPI', methods=['POST'])
def ChatGPTWebAPI():
    try:
        print("ChatGPTWebAPI request.data: ", request.data)
        prompt = json.loads(request.data)['prompt']
        completions = openai.Completion.create(
            engine=config.engine,
            prompt=prompt,
            max_tokens=config.max_tokens,
            top_p=config.top_p
        )
        message = completions.choices[0].text.strip()
        message=""
        query = classification.query_intent(prompt)
        # if entities == ""
        print("query: ", query)
        return jsonify({"message": message, "query":query})

    except openai.exceptions.OpenAIError as e:
        # Check if the exception was due to a timeout
        if "timeout" in str(e).lower():
            return jsonify({"message": config.timeOutErrorMessage})
        # For any other OpenAIError
        return jsonify({"message": config.anyOtherExceptionErrorMessage, "query":"Error, see logs for details."})
    except Exception as e:
        # For any other exception
        return jsonify({"message": config.anyOtherExceptionErrorMessage, "query":"Error, see logs for details."})

# UI FOR TESTING THE API
@app.route('/api/ChatGPTWebAPITester', methods=['GET', 'POST'])
def chatGPTWebAPITester():
    try:
        if request.method == 'POST':
            prompt = request.form['prompt']
            # # completions = openai.Completion.create(
            # #     engine=config.engine,
            # #     prompt=prompt,
            # #     max_tokens=config.max_tokens,
            # #     top_p=config.top_p
            # # )
            # # message = completions.choices[0].text.strip()
            # # message = helpers.remove_extra_emojis(message)
            message = "Dolphin AI Results:"

            # ( intent, entities, query) = classification.query_intent(prompt)
            ( intent, entities, query) = ("", "", "")
            print(query)
            print(prompt.lower())

            print("query: ", query, "intent: ", intent, "entities: ", entities)

            return render_template('ChatGPTWebAPITester.html',
                title='Direct Machines AI Chatbot Tester',
                year=datetime.now().year,
                message='Use /api/ChatGPTWebAPI for actual API calls.',
                response={"Hello": message, "query": query, "intent": intent, "entities": entities})
        else:
            return render_template('ChatGPTWebAPITester.html',
                title='Direct Machines AI Chatbot Tester',
                year=datetime.now().year,
                message='Use /api/ChatGPTWebAPI for actual API calls.')
    
    except openai.error as e:
        # Check if the exception was due to a timeout
        if "timeout" in str(e).lower():
            return jsonify({"message": config.timeOutErrorMessage})
        # For any other OpenAIError
        return jsonify({"message": config.anyOtherExceptionErrorMessage})
    except Exception as e:
        # For any other exception
        return jsonify({"message": config.anyOtherExceptionErrorMessage})

if __name__ == '__main__':
    socketio.run(app)
    

# if __name__ == '__main__':
#     app.run(debug=True, port=os.getenv("PORT", default=5000))

# defaults http://127.0.0.1:5000/api/ChatGPTWebAPITester