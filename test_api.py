from openai import OpenAI   # openai = 装好的工具箱（DeepSeek 兼容它，所以用它）

client = OpenAI(
    api_key="sk-0a015e865c06404cb7ef64bd4d4e2687",      # api_key = 你的门禁卡
    base_url="https://api.deepseek.com",        # base_url = 餐厅地址
)

response = client.chat.completions.create(
    model="deepseek-chat",                       # model = 点哪个厨师
    messages=[                                   # messages = 对话记录
        {"role": "user", "content": "用一句话介绍你自己"}   # role = 角色（user=用户），content = 内容
    ],
)

print(response.choices[0].message.content)       # 取出回答并打印
