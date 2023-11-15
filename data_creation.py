import replicate
import random
import logging
import json
import re
import os

# Configuration
# DO NOT CHANGE THESE PARAMETERS
MODEL_NAME = "mistralai/mistral-7b-v0.1:3e8a0fb6d7812ce30701ba597e5080689bef8a013e5c6a724fafb108cc2426a0"
MAX_NEW_TOKENS = 128
REPLICATE_API_TOKEN = 'r8_2exQ6CNhSRtslvEwof1vEMwXbx8RLr53VlRmP'

# YOU CAN CHANGE THESE PARAMETERS
NUM_CONVERSATIONS = 50
QUALITY_THRESHOLD = 1


# Initialize logging
logging.basicConfig(level=logging.INFO)

# Cache structure: {"conversation": String, "quality": Integer}
conversation_quality_cache = {}
conversations = []


def load_from_file(file_name):
    with open(file_name, 'r') as file:
        file_content = file.read()
        exec(file_content, globals())


# Load the lists from text files
load_from_file('inputs/scenarios.py')
load_from_file('inputs/tones.py')
load_from_file('inputs/partners.py')


def run_mistral(prompt, max_new_tokens=MAX_NEW_TOKENS):
    try:
        output = replicate.run(
            MODEL_NAME,
            input={"prompt": prompt,
                   "max_new_tokens": max_new_tokens, "temperature": 0.16}
        )
        # HINT: See if you want to change this to keep only the first output. What are the benefits / drawbacks?
        return "".join(list(output))
    except Exception as e:
        logging.error(f"API call failed with error: {e}")
        return ""


def check_conversation_quality(conversation):
    quality = conversation_quality_cache.get(conversation, {}).get("quality")
    raw_rating_response = ""

    if quality is None:

        quality_prompt = (f"Conversation for Rating:\n{conversation}\n\n---\n"
                          "As a neutral third-party agent designed to rate the quality of conversations "
                          "on a scale of 1 to 5, where 1 is very poor and 5 is excellent, with no further justification (I am very concise), where any conversation which is simply blank with nothing but USER and ASSISTANT tags is automatically a 1, the quality of this "
                          "conversation is ")
        rating = run_mistral(quality_prompt, 1)
        raw_rating_response = rating  # Save the raw response

        try:
            # Using regex to find the first number in the response
            score_match = re.search(r'\d+', rating)
            if score_match:
                score = int(score_match.group())
                quality = min(max(score, 1), 5)
            else:
                raise ValueError("No numeric score found in response.")
        except Exception as e:
            logging.error(f"Failed to parse quality score: {e}")
            quality = 3

        conversation_quality_cache[conversation] = {
            "quality": quality, "raw_response": raw_rating_response}

    return quality


def generate_conversation():
    # Randomly select elements from each list
    selected_partner = random.choice(random.choice(partners))
    selected_tone = random.choice(tones)
    selected_scenario, locations = random.choice(scenarios)
    location = random.choice(locations)

    # Constructing the detailed conversation prompt
    conversation_prompt = (f"Generate a conversation between a user and an AI assistant about '{selected_scenario}' at '{location}'. "
                           f"The user is a {selected_partner}, starting off in a '{selected_tone[0]}' tone and shifting to a '{selected_tone[1]}' tone. "
                           "The conversation should start with a relevant, specific question from the user for assistance from the assistant, "
                           "and follow the format of alternating messages between USER and ASSISTANT. "
                           "The conversation is output in this format\n"
                           "USER: [user's message]\n"
                           "ASSISTANT: [assistant's message]\n"
                           "The conversation can have 1-6 turns. Return the conversation below. Use English.\n"
                           "CONVERSATION:")

    # Generate the conversation
    conversation = run_mistral(conversation_prompt)

    print(conversation)
    # Store in cache (with quality uninitialized)
    conversation_quality_cache[conversation] = {}

    return conversation


def format_conversation(raw_conversation, max_turns=6):
    # Split the conversation into turns
    turns = raw_conversation.split("\n")

    # Keep only up to 6 turns
    if len(turns) > max_turns * 2:
        turns = turns[:max_turns * 2]

    # Rejoin the turns without adding tags
    formatted_conversation = " ".join(turns)

    # Add the optional system message
    system_message = "<SYS> Below are a series of dialogues between various people and an AI assistant. The AI tries to be helpful, polite, honest, sophisticated, emotionally aware, and humble-but-knowledgeable. </SYS>"
    formatted_conversation = f"{system_message} {formatted_conversation}"

    # Replace newline characters
    formatted_conversation = formatted_conversation.replace("\n", "\\n")

    return formatted_conversation


def main():
    logging.info("Generating conversations...")

    # Generate conversations
    conversations = []
    while len(conversations) < NUM_CONVERSATIONS + 1:
        conversation = generate_conversation()
        if check_conversation_quality(conversation) >= QUALITY_THRESHOLD:
            conversations.append(conversation)

            # Save raw conversations for inspection
    with open("raw_conversations.txt", "w") as raw_file:
        for conv in conversations:
            raw_file.write(conv.replace("\n", " ") + "\n")

    logging.info("Filtering by quality...")
    high_quality_conversations = conversations

    # Save the quality ratings
    with open("conversation_quality.json", "w") as quality_file:
        json.dump(conversation_quality_cache, quality_file, indent=4)

    # Save the raw rating responses along with the conversations
    with open("raw_rating_responses.txt", "w") as rating_file:
        for conv, data in conversation_quality_cache.items():
            formatted_conv = conv.replace("\n", "\\n")
            rating_file.write(
                f"Conversation: {formatted_conv}\nRaw Rating Response: {data.get('raw_response', '')}\n\n")

    # Format and save high-quality conversations
    with open("synthetic_dataset.txt", "w") as f:
        for conv in high_quality_conversations:
            conversation_formatted = format_conversation(conv)
            f.write(conversation_formatted + "\n")

    logging.info(
        f"Saved {len(high_quality_conversations)} high-quality conversations.")


if __name__ == "__main__":
    main()
