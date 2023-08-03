from gevent import monkey
monkey.patch_all()


from flask import Flask, jsonify, render_template, request, session
from flask_session import Session

import os
from datetime import datetime
from flask_socketio import SocketIO, emit
import classification, config
import openai
import json
from dotenv import load_dotenv
import time

# DREW HELPER FUNCTIONS
# from PyPDF2 import PdfReader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# from langchain.chat_models import ChatOpenAI
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
from embedder import get_vector_results

import io
import base64
# get the API key from the .env file

# Define a local file path to save the PDF
pdf_path = "temp_pdf_file.pdf"

# Delete the existing file if it exists
if os.path.exists(pdf_path):
    os.remove(pdf_path)


app = Flask(__name__)
app.secret_key = 'some_secret_string'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Session(app)
# socketio = SocketIO(app, async_mode='gevent')
socketio = SocketIO(app, async_mode='threading', engineio_logger=True)

pdf_path = None
messages = []

conversation_log = []
# socketio = SocketIO(app, cors_allowed_origins="*")


load_dotenv()


API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = API_KEY


for i in range(10):
    socketio.emit('response', {"message": i})  # send the chunk message and result_dict to the client

@app.route('/')
@app.route('/home')
def home():
    # set secret key
    # session['conversation'] = None
    # session['chat_history'] = None

    """Renders the home page."""
    return render_template(
        'index.html',
        title='Home Page',
        year=datetime.now().year,
    )


from flask import jsonify, request, session
import base64
import os

@app.route('/api/vectorize', methods=['POST'])
def vectorize():
    try:
        # Define a local file path to save the PDF
        pdf_path = "temp_pdf_file.pdf"
        
        # Delete the existing file if it exists
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

        base64_pdf = request.json['data']
        pdf_data = base64.b64decode(base64_pdf)
        
        with open(pdf_path, 'wb') as f:
            f.write(pdf_data)

        # Now, you can reference the PDF path in your get_vector_results function
        vector_results = get_vector_results('', pdf_path)  # Modify the function accordingly
        print("*******************************vector_results", vector_results)
        return jsonify({"status": "success", "files": vector_results})
    except Exception as e:
        print("*!*!*!*!*!*!*!*!*!*!*!*ERROR:", e)  # Print or log the error for debugging
        return jsonify({"status": "error", "message": str(e)})

@socketio.on('message')
def handle_message(data):
    global messages
    prompt = data['prompt']
    temperature = config.top_p

    pdf_path = "temp_pdf_file.pdf"  # Same path as in vectorize function
    if os.path.exists(pdf_path):
        prescript_message = {"role": "user", "content": "This is the results based on a PDF, which may or may not be helpful: \n" + get_vector_results(prompt, pdf_path)[0].page_content}
        print("*******************************prescript_message", prescript_message)
        messages.append(prescript_message)
    else:
        print("*******************************pdf_path does not exist")
    # Add the new user message to the messages list.
    messages.append({"role": "user", "content": prompt})
    
    # Keep only the last 5 messages.
    messages = messages[-5:]

    # result_dict = classification.query_intent(prompt)
    result_dict = {"squads": {}, "results": [], "response": ""}
    if result_dict['squads'] != {} or result_dict['results'] != []:
        prompt = prompt + config.foundResults + f'{result_dict}'

    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=messages,
        temperature=temperature,
        stream=True
    )
    full_response = ""
    for chunk in response:
        chunk_message = chunk['choices'][0]['delta']
        # Add the new assistant message to the messages list.
        try: 
            full_response += chunk_message['content']
        except:
            print("No content in chunk message", chunk_message)
        socketio.emit('response', {"message": chunk_message})
    messages.append({"role": "assistant", "content": full_response})
    # print("VECTOR",get_vector_results(full_response, 'doall.pdf'))
    # Keep only the last 5 messages.
    messages = messages[-5:]

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
        # query = classification.query_intent(prompt)
        query = "SELECT * FROM table WHERE entities = 'entities'"
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
        pdf_path = None
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
    
    # except openai.exceptions.OpenAIError as e:
    #     # Check if the exception was due to a timeout
    #     if "timeout" in str(e).lower():
    #         return jsonify({"message": config.timeOutErrorMessage})
    #     # For any other OpenAIError
    #     return jsonify({"message": config.anyOtherExceptionErrorMessage})
    except Exception as e:
        # For any other exception
        print(e)
        return jsonify({"message": config.anyOtherExceptionErrorMessage})

if __name__ == '__main__':
    # socketio.run(app)
    app.run(host='0.0.0.0', port=5555)


# if __name__ == '__main__':
#     app.run(debug=True, port=os.getenv("PORT", default=5000))

# defaults http://127.0.0.1:5000/api/ChatGPTWebAPITester



# @socketio.on('message')
# def handle_message(data):
#     full_message = ''
#     load_dotenv()
#     API_KEY = os.getenv("OPENAI_API_KEY")
#     print(API_KEY)
#     openai.api_key = API_KEY  # ensure API key is set locally
#     conversation_log = session.get('conversation_log', [])
    
#     if len(conversation_log) < 2:
#         prompt = "You are Machinelle, a store clerk for Direct Machines, a metal machinery ecommerce retailer. Please handle the following customer request, being sure to be as helpful as possible:\n"+ data['prompt']
#     else:
#         prompt = data['prompt']
#     temperature = .8
#     result_dict = classification.query_intent(data['prompt'])
#     # intent, entities, query, results = result_dict
#     # NOW RUN THE PROMPT:
#     print("Result dict: ", result_dict)
#     # Update the conversation log before responding
#     conversation_log.append({"role": "user", "content": prompt})
#     if len(conversation_log) > 5:
#         conversation_log.pop(0)  # remove the oldest message if more than 5 messages

#     # socketio.emit('response', {"message":"", "intent": intent, "entities": [(f" {entity}: {entities[entity]}") for entity in entities], "query": query, "results": results}) 
#     print(conversation_log)
#     response = openai.ChatCompletion.create(
#         model='gpt-4',
#         messages=conversation_log,
#         temperature=temperature,
#         stream=True,  # set stream=True
#         max_tokens=250
#     )
#     for chunk in response:
#         time.sleep(1)

#         chunk_message = chunk['choices'][0]['delta']  # extract the message
#         print("CHUNKY",chunk_message)
#         socketio.emit('response', {"message": chunk_message})  # send the chunk message and result_dict to the client
#         try:
#             full_message += chunk_message["content"]
#         except:
#             continue # if the message is not a string, skip it
#     if len(conversation_log) > 5:
#         conversation_log.pop(0)  # remove the oldest message if more than 5 messages
#     conversation_log.append({"role": "assistant", "content": full_message})

#     session['conversation_log'] = conversation_log
