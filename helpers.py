#helpers.py
import config
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
