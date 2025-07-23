from datasets import load_dataset

dataset = load_dataset("silk-road/ChatHaruhi-54K-Role-Playing-Dialogue")

role2tuple = {}

from tqdm import tqdm
for data in tqdm( dataset['train'] ):
    agent_role = data['agent_role']
    if agent_role not in role2tuple:
        role2tuple[agent_role] = []

    user_role = data['user_role']
    user_question = data['user_question']

    query = user_role + ":" + user_question + ""

    agent_response = data['agent_response']

    target = agent_role + ":" + agent_response + ""

    # role2tuple[agent_role].append((query, target))

    history_str = ""

    tmp_data = {
        "query": query,
        "target": target,
        "history_str": history_str,
    }

    role2tuple[agent_role].append(tmp_data)

    more_dialogues = data['more_dialogues']

    history_str += query + "\n" + target

    history_prefix = history_str

    if len(more_dialogues) > 0:
        n = len( more_dialogues )
        for i in range(n):
            if i == 0:
                continue

            sent = more_dialogues[i]
            if len(sent) < len(agent_role):
                continue

            if sent.startswith(agent_role):
                query = more_dialogues[i-1]
                target = sent

                history_str = history_prefix
                for j in range(i-1):
                    history_str += "\n" + more_dialogues[j]

                tmp_data = {
                    "query": query,
                    "target": target,
                    "history_str": history_str,
                }

                role2tuple[agent_role].append(tmp_data)

                # print(tuple_data)

                # break

        # break

K_SEARCH = 5
MAX_LEN_STORY = 1000 #这个是按照token算的
MAX_LEN_HISTORY = 1200 # count with token

import os
import openai
key = "key is not necessary here"
key_bytes = key.encode()
os.environ["OPENAI_API_KEY"] = key_bytes.decode('utf-8')

from ChatHaruhi import ChatHaruhi

from ChatHaruhi.ChromaDB import ChromaDB

class ChatHaruhiTrain(ChatHaruhi):

    def build_story_db_from_vec( self, texts, vecs ):
        self.db = ChromaDB()
        self.stories = texts
        self.db.init_from_docs( vecs, texts)

    def add_story_with_expire(self, query, expire_story):
        if self.db is None:
            print("No vec DB！")
            return

        query_vec = self.embedding(query)
        stories = self.db.search(query_vec, self.k_search)

        story_string = self.story_prefix_prompt + self.dialogue_divide_token

        sum_story_token = self.tokenizer(story_string)

        for story in stories:
            if expire_story is not None and expire_story.strip() == story.strip():
                continue

            if expire_story == None and query.strip() in story.strip():
                continue

            story_token = self.tokenizer(story.strip()) + self.tokenizer(self.dialogue_divide_token)
            if sum_story_token + story_token > self.max_len_story:
                break
            else:
                sum_story_token += story_token
                story_string += story.strip() + self.dialogue_divide_token

        self.llm.user_message(story_string)

    def generate_prompt(self, query, history_str, expire_story ):
        # 这里修改下其他超参，不规范删了
        # self.k_search = 5
        # self.max_len_story = 1500
        # self.max_len_history = 1200
        self.story_prefix_prompt = "\nClassic scenes for the role are as follows:"

        self.llm.initialize_message()

        self.llm.system_message(self.system_prompt)

        self.add_story_with_expire(query, expire_story)

        # self.add_history(history)
        history_message = self.dialogue_divide_token + history_str.strip()
        self.llm.user_message(history)

        self.llm.user_message(query)

        # self.llm.user_message(target)

        return self.llm.messages

    def add_history(self, history_list):

        if len(history_list) == 0:
            return

        sum_history_token = 0
        flag = 0
        for history in history_list:
            current_count = 0
            if history is not None:
                current_count += self.tokenizer(history)

            sum_history_token += current_count
            if sum_history_token > self.max_len_history:
                break
            else:
                flag += 1

        if flag == 0:
            print('warning! no history added. the last dialogue is too long.')

        # 是否添加历史前缀，
        history_message = ""
        for history in history_list[-flag:]:
            history_message += history
        self.llm.user_message(history_message)

role_name_Haruhiu = {'凉宫春日': 'haruhi', 'haruhi': 'haruhi', 'Haruhi': 'haruhi', '春日': 'haruhi'}

count = 0

for role in role2tuple:
    if role in role_name_Haruhiu:
        for ele in role2tuple[role]:
            if isinstance( ele, dict):
                count += 1

print(count)

import zipfile
import os

try:
    os.makedirs("characters_zip")
except:
    pass
try:
    os.makedirs("characters")
except:
    pass

role_en2bots = {}


from tqdm import tqdm

for ai_role_en in tqdm( role_name_Haruhiu.values() ):
    if ai_role_en in role_en2bots:
        continue

    role_en2bots[ai_role_en] = ChatHaruhiTrain(role_name = ai_role_en)

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from datasets import load_dataset
from tqdm import tqdm

save_datas = []

# for role_name in tqdm(role_from_roleLLM):

for role_name_zh in role2tuple.keys():
    if role_name_zh not in role_name_Haruhiu:
        continue
    role_name_en = role_name_Haruhiu[role_name_zh]

    chatbot = role_en2bots[role_name_en]


    chatbot.k_search = K_SEARCH
    chatbot.max_len_story = MAX_LEN_STORY
    chatbot.max_len_history = MAX_LEN_HISTORY

    all_tuples = role2tuple[role_name_zh]

    # break

    for tuple_data in tqdm(all_tuples):
        query = tuple_data['query']
        target = tuple_data['target']
        history = tuple_data['history_str']
        messages = chatbot.generate_prompt(query, history, None)
        # print(prompt)

        prompt = ""

        for msg in messages:
            if isinstance(msg, HumanMessage):
                prompt += msg.content + "\n"
            elif isinstance(msg, AIMessage):
                prompt += msg.content + "\n"
            elif isinstance(msg, SystemMessage):
                prompt += msg.content + "\n"

        save_data = {
            'context': prompt,
            'target': target
        }

        save_datas.append(save_data)

        # break

    # break

import json

save_name = "./ChatHaruhi_54K_retide.jsonl"

with open(save_name, 'w', encoding='utf8') as f:
    for data in save_datas:
        json.dump(data, f, ensure_ascii=False)
        f.write('\n')