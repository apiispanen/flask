# -*- coding: utf-8 -*-
from datetime import datetime
from flask import render_template, jsonify, request
from flask import Response, stream_with_context
from SpinnrAIWebService import app, apiconfig, config, helpers, transcriber #, classification
import openai, json
from flask_socketio import SocketIO, emit
import time

socketio = SocketIO(app, async_mode='threading', engineio_logger=True)
# socketio = SocketIO(app, async_mode='gevent', engineio_logger=True)
# socketio = SocketIO(app, cors_allowed_origins="*")

messages = []
openai.api_key = apiconfig.API_KEY

@app.route('/')
@app.route('/home')
def home():
    """Renders the home page."""
    return render_template(
        'index.html',
        title='Home Page',
        year=datetime.now().year,
    )

@app.route('/api/moderate', methods=['POST', 'GET'])
def moderate():
    # RETURNS A FLAG FOR ANY CONTENT IN THE MESSAGE THAT HAS OFFENSIVE LANGUAGE
    text = json.loads(request.data)['text']
    flagged = transcriber.moderate_text(text)
    return jsonify({"flagged_content": flagged})

@app.route('/api/transcribe', methods=['POST', 'GET'])
def transcribe():
    # RETURNS THE TRANSCRIPT OF A VIDEO OR AUDIO FILE
    url = json.loads(request.data)['url']
    transcript = transcriber.get_transcription(url)
    return jsonify({"transcript": transcript})

# @app.route('/api/classify', methods=['POST', 'GET'])
# def classify():
#     # CLASSIFY IS USED TO CLASSIFY THE INTENT OF THE USER'S REQUEST
#     prompt = json.loads(request.data)['prompt']
#     print("PROMPT",prompt)
#     if prompt is None:
#         return jsonify({"error": "Invalid JSON"}), 400
#     # prompt = json.loads(request.data)['prompt']
#     prompt = helpers.AddEmojiRequestToPrompt(prompt)
#     if "spinnr" in prompt.lower():
#         return jsonify({"message": config.spinnrQuestionMessage})
#     elif "spinny" in prompt.lower():
#         prompt = prompt.replace("spinny", "")

#     result_dict = classification.query_intent(prompt)
#     if result_dict['squads'] != {} or result_dict['results'] != []:
#         prompt = prompt + config.foundResults + f'{result_dict}'

#     print("FULL PROMPT TO GPT: ", prompt)
#     messages = [{"role": "user", "content": prompt}]
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo-0301",
#         messages=messages,
#         max_tokens=config.max_tokens,
#         n=1,
#         stop=None,
#         temperature=config.top_p,
#         frequency_penalty=1,
#         stream=False
#     )
#     print("RESPONSE FROM GPT: ", response)
#     return jsonify({"message": response.choices[0].text, "result_dict": result_dict})

@app.route('/api/ChatGPTWebAPIStream', methods=['POST', 'GET'])
def ChatGPTWebAPIStream():
    data = request.get_json()
    if data is None:
        return jsonify({"error": "Invalid JSON"}), 400
    prompt = data.get('prompt')
    if prompt is None:
        return jsonify({"error": "No prompt provided"}), 400
    # prompt = json.loads(request.data)['prompt']
    prompt = helpers.AddEmojiRequestToPrompt(prompt)
    if "spinnr" in prompt.lower():
        return jsonify({"message": config.spinnrQuestionMessage})
    elif "spinny" in prompt.lower():
        prompt = prompt.replace("spinny", "")

    # result_dict = classification.query_intent(prompt)
    result_dict = {"squads": {}, "results": [], "response": ""}
    if result_dict['squads'] != {} or result_dict['results'] != []:
        prompt = prompt + config.foundResults + f'{result_dict}'

    print("FULL PROMPT TO GPT: ", prompt)
    messages = [{"role": "user", "content": prompt}]
    def generate():
        response_text = ""
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0301",
            messages=messages,
            max_tokens=config.max_tokens,
            n=1,
            stop=None,
            temperature=config.top_p,
            frequency_penalty=1,
            stream=True
        )

        for chunk in response:
            chunk_message = chunk['choices'][0]['delta']
            print("Chunk message:", chunk_message)
            socketio.emit('response', chunk_message)
            try:
                response_text += chunk_message['content']
            except:
                print("No content in chunk message", chunk_message)
            
            # Convert chunk_message to a JSON string and then encode it into bytes
            yield json.dumps(chunk_message).encode('utf-8')

        # set response to the entire response object
        result_dict['response'] = response_text
        print("result_dict: ", result_dict)
        
        # Convert the result dictionary to a JSON string and then encode it into bytes
        result_json = json.dumps(result_dict)
        return result_json.encode('utf-8')

    return Response(generate(), mimetype='text/event-stream')

# Static Response
@app.route('/api/ChatGPTWebAPI', methods=['POST', 'GET'])
def ChatGPTWebAPI():
    prompt = json.loads(request.data)['prompt']
    prompt = helpers.AddEmojiRequestToPrompt(prompt)
    if "spinnr" in prompt.lower():
        return jsonify({"message": config.spinnrQuestionMessage})
    elif "spinny" in prompt.lower():
        prompt = prompt.replace("spinny", "")
    # result_dict = classification.query_intent(prompt)
    result_dict = {"squads": {}, "results": [], "response": ""}
    
    if result_dict['squads'] != {} or result_dict['results'] != []:
        prompt = prompt + config.foundResults + f'{result_dict}'

    print("FULL PROMPT TO GPT: ", prompt)
    messages = [{"role": "user", "content": prompt}]
    response_text = ""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        max_tokens=config.max_tokens,
        n=1,
        stop=None,
        temperature=config.top_p,
        frequency_penalty=1,
        stream=False
    )
    print("response: ", response)
    response_message = response["choices"][0]["message"]["content"].strip()
    print("response_message :", response_message)
    # set response to the entire response object

    result_dict['response'] = response_message
    print("result_dict: ", result_dict)
    return result_dict

@app.route('/api/ChatGPTWebAPITester', methods=['GET', 'POST'])
def chatGPTWebAPITester():
    try:
        if request.method == 'POST':
            prompt = request.form['prompt']
            if "spinnr" in prompt.lower():
                return jsonify({"message": config.spinnrQuestionMessage})
            elif "spinny" in prompt.lower():
                prompt = prompt.replace("spinny", "")
            prompt = helpers.AddEmojiRequestToPrompt(prompt)
            completions = openai.Completion.create(
                engine=config.engine,
                prompt=prompt,
                max_tokens=config.max_tokens,
                top_p=config.top_p
            )
            message = completions.choices[0].text.strip()
            message = helpers.remove_extra_emojis(message)
        
            return render_template('ChatGPTWebAPITester.html',
                title='ChatGPT Tester',
                year=datetime.now().year,
                message='Use /api/ChatGPTWebAPI for actual API calls.',
                response={"Hello": message})
        else:
            return render_template('ChatGPTWebAPITester.html',
                title='ChatGPT Tester',
                year=datetime.now().year,
                message='Use /api/ChatGPTWebAPI for actual API calls.')
    
    except Exception as e:
        # For any other exception
        print(e)
        return jsonify({"message": config.anyOtherExceptionErrorMessage})

@app.route('/api/VideoIntelligenceAPITester',  methods=['GET', 'POST'])
def VideoIntelligenceAPITester():
    try:
        if request.method == 'POST':
            path = request.form['path']
            content_analysis_dict = helpers.analyze_explicit_content(path)
            return render_template('VideoIntelligenceAPITester.html',
                title='Video Intelligence API Tester',
                year=datetime.now().year,
                message='Use /api/VideoIntelligenceAPI for actual API calls.',
                response=content_analysis_dict)
        else:
            return render_template('VideoIntelligenceAPITester.html',
                title='Video Intelligence API Tester',
                year=datetime.now().year,
                message='Use /api/VideoIntelligenceAPI for actual API calls.')
    except Exception as e:
        # For any other exception
        return jsonify({"message": config.anyOtherExceptionErrorMessage})

@socketio.on('message')
def handle_message(data):
    global messages
    prompt = data['prompt']
    temperature = config.top_p

    # Add the new user message to the messages list.
    messages.append({"role": "user", "content": prompt})
    
    # Keep only the last 5 messages.
    messages = messages[-5:]

    # result_dict = classification.query_intent(prompt)
    result_dict = {"squads": {}, "results": [], "response": ""}
    if result_dict['squads'] != {} or result_dict['results'] != []:
        prompt = prompt + config.foundResults + f'{result_dict}'

    response = openai.ChatCompletion.create(
        model='gpt-4',
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
    # Keep only the last 5 messages.
    messages = messages[-5:]

if __name__ == '__main__':
    socketio.run(app)

# THE FUNCTIONS SHOULD:
# 1. Take in a prompt, understand intents, and return a response
#       Events
#       Squads
#       Users
# 2. Give a welcome message

