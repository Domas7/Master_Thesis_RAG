from openai import OpenAI
from private_key import get_api_keys



def get_chat_completion(instructions, specifications):
    priv = get_api_keys()
    client = OpenAI(api_key=priv)
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": instructions
            },

            {
                "role": "user",
                "content": specifications
            }
        ]
    )

    print(f"AI completion: {completion.choices[0].message.content}")
    return completion.choices[0].message.content


def get_chat_completion1(userinput):
    instructions = """

    This is the user input: I want to have to see lessons learned from mission 5402
    This is what you should reply: /lessons-learned 5402

    This is the user input: I would like to see summary of the lessons learned from mission 127, 58, 3020
    This is what you should reply: /summary 127, 58, 3020

    If the userinput does not match any of the intended functions or questions, you should respond with "command not found or no data was found in the database".
    """

    specifications = f"""
    When writing a command the command should not have a space between the / and the command word. Ie. the command should be written as /add or /list
    This is the user input that you should interpret: {userinput}
    """

    priv = get_api_keys()
    client = OpenAI(api_key=priv)
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": instructions
            },

            {
                "role": "user",
                "content": specifications
            }
        ]
    )

    print(f"AI completion: {completion.choices[0].message.content}")

    return completion.choices[0].message.content