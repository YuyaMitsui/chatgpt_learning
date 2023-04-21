import openai

prompt = "What is ChatGPT?"

res = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": f"res: japanese \n {prompt} "
        }
    ]
)

content = res.choices[0].message.content
print(content)
