#helpers.py
from SpinnrAIWebService import config, apiconfig
from google.cloud import videointelligence
import re
import io

def AddEmojiRequestToPrompt(prompt):
    prompt = prompt + config.appendEmojiRequest
    return prompt

def remove_extra_emojis(api_response: str) -> str:
    # Split the API response into paragraphs
    paragraphs = api_response.splitlines()

    # Iterate through each paragraph
    for i in range(len(paragraphs)):
        # Use regular expressions to match emojis in the paragraph
        matches = re.findall(r"(?:[\u2700-\u27bf]|(?:\ud83c[\udde6-\uddff]){2}|[\ud800-\udbff][\udc00-\udfff]|(\u00a9|\u00ae|[\u2000-\u3300] |\ud83c[\ud000-\udfff]|\ud83d[\ud000-\udfff]|\ud83e[\ud000-\udfff]))", paragraphs[i])

        # If there are more than one emojis in the paragraph, remove all emojis after the first one
        if len(matches) > 1:
            for j in range(1, len(matches)):
                paragraphs[i] = paragraphs[i].replace(matches[j], "")

    # Rejoin the paragraphs into a single string
    api_response = "\n".join(paragraphs)

    return api_response

def analyze_explicit_content(path):
    # [START video_analyze_explicit_content]
    """Detects explicit content from the GCS path to a video."""
    video_client = videointelligence.VideoIntelligenceServiceClient()
    features = [videointelligence.Feature.EXPLICIT_CONTENT_DETECTION]

    operation = video_client.annotate_video(
        request={"features": features, "input_uri": path}
    )   
    print("\nProcessing video for explicit content annotations:")

    result = operation.result(timeout=90)
    print("\nFinished processing.")

    content_analysis_dict = {}
    # Retrieve first result because a single video was processed
    for frame in result.annotation_results[0].explicit_annotation.frames:
        likelihood = videointelligence.Likelihood(frame.pornography_likelihood)
        frame_time = frame.time_offset.seconds + frame.time_offset.microseconds / 1e6
        content_analysis_dict[frame_time] = likelihood.name
        print("Time: {}s".format(frame_time))
        print("\tpornography: {}".format(likelihood.name))

    return content_analysis_dict

def get_user_summary(guid, username):
    import requests
    data = {"guid":guid,"username":username}
    url = 'https://spinnrdev.com/getUserprofile.php'
    r = requests.post(url, data=data)
    print(r.text)
    return r.text


# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.chat_models import ChatOpenAI
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# def ai_response(prompt):
#     def get_vectorstore(text_chunks):
#         embeddings = OpenAIEmbeddings(openai_api_key=apiconfig.API_KEY)
#         # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
#         vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
#         return vectorstore


#     def get_conversation_chain(vectorstore):
#         llm = ChatOpenAI(openai_api_key=apiconfig.API_KEY)
#         # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
#         memory = ConversationBufferMemory(
#             memory_key='chat_history', return_messages=True)
#         conversation_chain = ConversationalRetrievalChain.from_llm(
#             llm=llm,
#             retriever=vectorstore.as_retriever(),
#             memory=memory
#         )
#         return conversation_chain

#     get_vectorstore(prompt)
#     vectorstore = get_vectorstore(prompt)
#     conversation_chain = get_conversation_chain(vectorstore)
#     print(conversation_chain)
