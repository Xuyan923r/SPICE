from openai import OpenAI

client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role":"user", "content":"hello"}]
)

print(resp.choices[0].message.content)
