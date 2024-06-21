from openai import OpenAI 

client = OpenAI(
    api_key="c05fa59a1e2b3a3153dad34157a9f374.2fg9qbnQOLuiVvBI",
    base_url="https://open.bigmodel.cn/api/paas/v4/"
) 

if __name__ == "__main__":
    chat_completion = client.chat.completions.create(
        model="glm-4",
        messages=[
            {
                "role": "user",
                "content": "who are you"
            }
            ],
            temperature=0.9
    )
    print(chat_completion.choices[0].message)

    embed_com = client.embeddings.create(
        model="embedding-2",
        input="你好"
    )
    print(type(embed_com.data[0]))