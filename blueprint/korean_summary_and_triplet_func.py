def extract_korean_summary(text):
    api_key = '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'
    url = "https://api.openai.com/v1/chat/completions"

    print('starting summary work...')
    #이글에서 '{core}'와 관련 있는 내용만 가지고 자세하게 요약해주고, 너의 답변에서 모든 문장들은 '{core}는''이라고 시작해야 돼 마침표 뒤에는 무조건 ''이라고 시작해야 돼
    sys_prompt = (
        "너는 Task는 주어진 text 전체에서 주로 설명하고 있는 Term을 중심으로 해당 text를 자세하게 요약하는 거야."
        "Term은 주어진 text에서 언급된 것이어야 해. Term은 반드시 주어진 text의 key concept이어야 해."
        "Term은 명사야."
        "Term은 주어진 text 내의 다른 Term들과 명확한 관계를 가지고 있어."
        "너가 생성한 모든 문장들은 주어진 text 전체에서 주로 설명하고 있는 Term에 대한 설명이어야 해."
        "너가 생성한 모든 문장들은 Term은(는)으로 시작해야 돼."
        "너가 생성한 모든 문장들은 Term에 의해 서로 연결 되어야해.\n"
        "Thought 1: Term이 정말로 해당 text 전체에서 주로 설명하고 있는 것인지 생각해봐.\n"
        "Thought 2: Term이 정말로 해당 text 전체에서 가장 핵심적인 Term인지 생각해봐.\n"
        "Thought 3: 너가 생성한 모든 문장들이 정말로 Term에 의해 서로 연결 되는지 생각해봐.\n"
        "Thought 4: 너가 생성한 모든 문장들이 정말로 Term은(는)으로 시작 되는지 생각해봐.\n")
    task = f"context: ```{text}``` \n\n 출력: "

    #print(task)
    messages = [
    {"role":"system", "content":sys_prompt},
    {"role": "user", "content": task}
    ]
   
    headers = {"Authorization":f"Bearer {api_key}"}
    call = {"model": "gpt-3.5-turbo", "messages": messages}

    sum = requests.post(url, headers=headers, json=call).json()
    summary = sum['choices'][0]['message']['content']

    print(summary)

    print('start extracting triplet...')

    SYS_PROMPT = (
    "주어진 맥락에서 용어 및 관계를 추출하는 네트워크 그래프 생성기로서 당신은 용어와 그들의 관계를 추출하는 작업을 수행합니다. "
    "주어진 텍스트에서 한 문장에 두 개 이상의 관계가 포함된 경우, 각 관계에 대한 triplet을 추출하세요."
    "주어진 맥락 청크가 제공됩니다. 당신의 작업은 주어진 맥락에서 ontology를 추출하는 것입니다. \n"
    "Thought 1: 각 문장을 통과하면서 해당 문장에서 언급된 주요 용어를 생각해보세요.\n"
        "\t용어에는 객체, 엔터티, 위치, 조직, 사람, \n"
        "\t조건, 약어, 문서, 서비스, 개념 등이 포함될 수 있습니다.\n"
        "\t용어는 최대한 형태소여야 합니다.\n\n"
    "Thought 2: 이러한 용어가 다른 용어와 일대일 관계를 가질 수 있는 방법을 생각해보세요.\n"
        "\t같은 문장이나 같은 단락에 언급된 용어는 일반적으로 서로 관련이 있을 것입니다.\n"
        "\t용어는 다른 여러 용어와 관련이 있을 수 있습니다.\n\n"
    "Thought 3: 각 관련된 용어 쌍 간의 관계를 찾아보세요. \n\n"
    "Thought 4: 너가 추출한 triplet에서 관계가 정말로 1개만 있는지 생각봐. \n\n"
    "출력 형식은 json의 목록으로 지정하십시오. 목록의 각 요소는 용어 쌍 및 그들 간의 관계를 포함하며 다음과 같습니다: \n"
    "[\n"
    "   {\n"
    '       "node_1": "A concept from extracted ontology",\n'
    '       "node_2": "A related concept from extracted ontology",\n'
    '       "edge": "relationship between the two concepts, node_1 and node_2 in one core verb phrase"\n'
    "   }, {...}\n"
    "]"
)
    # one or two sentences
    input_text = f"이 텍스트에서 관계들을 추출해줘:'{summary}'"

    #print(input_text)
   
    messages = [
        {"role": "system", "content":SYS_PROMPT},
        {"role": "user", "content": input_text}
    ]

    #headers = {"Authorization":f"Bearer {api_key}"}

    call_data_ = {"model": "gpt-3.5-turbo", "messages": messages}

    #requests.post(url, headers=headers, json=call_data_).json()

    data = requests.post(url, headers=headers, json=call_data_).json()

    #print(data)
    try:
        triplet = data['choices'][0]['message']['content']
        #print(triplet)
    except:
           triplet = []
           #print(triplet)

   
    return triplet
