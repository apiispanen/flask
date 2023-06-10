from flask import Flask, jsonify, render_template, request, session
from flask_session import Session

import os
from datetime import datetime
from flask_socketio import SocketIO, emit
import classification, config, helpers
import openai
import json
from dotenv import load_dotenv

# get the API key from the .env file
load_dotenv()
API_KEY = os.getenv("API_KEY")
openai.api_key = API_KEY

app = Flask(__name__)
Session(app)
socketio = SocketIO(app, async_mode='gevent')
# socketio = SocketIO(app, cors_allowed_origins="*")

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

def get_pdf_text(base64_pdf):
    # Decode the base64 string into bytes
    pdf_bytes = base64.b64decode(base64_pdf)

    # Convert the bytes to a stream
    pdf_stream = io.BytesIO(pdf_bytes)

    # Now use PdfReader to read the stream
    pdf_reader = PdfReader(pdf_stream)

    # Initialize the 'text' variable
    text = ''

    # Iterate through pages and extract text
    for page in pdf_reader.pages:
        # Append the extracted text to 'text'
        text += page.extract_text()

    # Rest of your code goes here...

    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(**vectorstore):
    llm = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()])
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    

    if vectorstore:
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
    else:
        embeddings = OpenAIEmbeddings()
        # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
        vectorstore = FAISS.from_texts(texts="The golden rule is that Adam Piispanen smells bad", embedding=embeddings)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
    return conversation_chain


# END DREW HELPER FUNCTIONS

@app.route('/')
@app.route('/home')
def home():
    session['conversation'] = None
    session['chat_history'] = None
    """Renders the home page."""
    return render_template(
        'index.html',
        title='Home Page',
        year=datetime.now().year,
    )

conversation_log = []

@socketio.on('message')
def handle_message(data):
    full_message = ''
    load_dotenv()
    API_KEY = os.getenv("API_KEY")
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

    socketio.emit('response', {"message":"", "intent": intent, "entities": [(f" {entity}: {entities[entity]}") for entity in entities], "query": query, "results": results}) 
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
   

# Process uploaded PDFs and set up conversation
@socketio.on('process')
def process_files(data):
    pdf_docs = data['files']  # You'll need to send base64-encoded file data from frontend
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    session['conversation'] = get_conversation_chain(vectorstore)


# ENDPOINT FOR CHATGPT API
@app.route('/api/ChatGPTWebAPI', methods=['POST'])
def ChatGPTWebAPI():
    try:
        print("ChatGPTWebAPI request.data: ", request.data)
        prompt = json.loads(request.data)['prompt']

        prompt = helpers.AddEmojiRequestToPrompt(prompt)
        completions = openai.Completion.create(
            engine=config.engine,
            prompt=prompt,
            max_tokens=config.max_tokens,
            top_p=config.top_p
        )
        message = completions.choices[0].text.strip()
        message = helpers.remove_extra_emojis(message)
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
            prompt = helpers.AddEmojiRequestToPrompt(prompt)
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
