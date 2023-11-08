prompt_ = " "

import openai
import requests

api_key = '  '

url = "https://api.openai.com/v1/chat/completions"

core_ = f"이 문장에서 가장 핵심 Entity가 뭔지 알려줘. 답변은 가장 핵심 Entity로만 해줘:'{prompt_}'"

messages = messages = [
{"role": "user", "content": core_}
]

headers = {"Authorization":f"Bearer {api_key}"}

data_ = {"model": "gpt-3.5-turbo", "messages": messages}

requests.post(url, headers=headers, json=data).json()

data_ = requests.post(url, headers=headers, json=data_).json()

core = data_['choices'][0]['message']['content']

input_text = f"이 문장을 ('{core}', predicate, Entity) 형식으로 관계를 추출해줘. 이때 predicate는 기본형으로 해줘. 답변은 (Entity, predicate, Entity)로만 해줘: '{prompt_}'"

messages = messages = [
{"role": "user", "content": input_text}
]

headers = {"Authorization":f"Bearer {api_key}"}

data = {"model": "gpt-3.5-turbo", "messages": messages}

requests.post(url, headers=headers, json=data).json()

data = requests.post(url, headers=headers, json=data).json()

print(core)
print(data['choices'][0]['message']['content'])
