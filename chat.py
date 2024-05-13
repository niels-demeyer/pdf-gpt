# Example: reuse your existing OpenAI setup
from openai import OpenAI

# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")


def chat():
    content_user = input("Enter your question: ")
    print("You entered: ", content_user)
    print("Processing...")
    completion = client.chat.completions.create(
        model="local-model",  # this field is currently unused
        messages=[
            {
                "role": "system",
                "content": "You are a senior software engineer at a tech company. You are very experienced and have a lot of knowledge about software development. You are also very passionate about your work. You are a very friendly and helpful person. You are very patient and understanding. You are very good at explaining things to others. You are very good at solving problems",
            },
            {
                "role": "user",
                "content": content_user,
            },
        ],
        temperature=0.7,
    )

    print(f"Answer: {completion.choices[0].message.content}")
    return content_user  # return the user's input


# runs in a loop until the user presses "q" or Ctrl+C
def main():
    while True:
        try:
            content_user = chat()
            if content_user.lower() == "q":
                break
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
