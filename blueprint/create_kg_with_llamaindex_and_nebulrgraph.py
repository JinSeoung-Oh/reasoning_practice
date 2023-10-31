import os
import random
import json
import hashlib
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

##### Data preparation
validation_data, test_data = load_dataset("suolyer/pile_wikipedia", split=['validation', 'test'])

data = []
random_rows = random.sample(range(len(test_data)), 10)
build_data = [test_data[val]['text'] for val in random_rows]

m = hashlib.md5()
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def bert_len(text):
    tokens = tokenizer.encode(text)
    return len(tokens)

def create_chunk_dataset(content):
      m.update(content.encode('utf-8'))
      uid = m.hexdigest()[:12]
      text_splitter = RecursiveCharacterTextSplitter(
          chunk_size = 400,
          chunk_overlap  = 40,
          length_function = bert_len,
          separators=['\n\n', '\n', ' ', ''],
      )
      chunks = text_splitter.split_text(content)
      for i, chunk in enumerate(chunks):
          data.append({
              'id': f'{uid}-{i}',
              'text': chunk
          })

for dt in build_data:
    create_chunk_dataset(dt)

filename = './kg/data/knowledge graphs/rebel_llamaindex/wiki_chunks.json'
# save
with open(filename, 'w') as outfile:
    for x in data:
        outfile.write(json.dumps(x) + '\n')


##### REBEL model for triplets
def extract_triplets(text):
    triplets = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    #print(text)
    #print(text.replace("<s>", "").replace("<pad>", "").replace("</s>", ""))
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        #print(token)
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
    return triplets

tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")

gen_kwargs = {
    "max_length": 256,
    "length_penalty": 0,
    "num_beams": 3,
    "num_return_sequences": 1,
}

triples = []

def generate_triples(texts):

  model_inputs = tokenizer(texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
  generated_tokens = model.generate(
      model_inputs["input_ids"].to(model.device),
      attention_mask=model_inputs["attention_mask"].to(model.device),
      **gen_kwargs
  )
  decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
  for idx, sentence in enumerate(decoded_preds):
      #print(sentence)
      et = extract_triplets(sentence)
      for t in et:
        triples.append((t['head'], t['type'], t['tail']))

for i in tqdm(range(0, len(data), 2)):
  try:
    texts = [data[i]['text'], data[i+1]['text']]
  except:
    texts = [data[i]['text']]
  #print(texts)
  generate_triples(texts)

distinct_triples = list(set(triples))

# save
with open('./kg/data/knowledge graphs/rebel_llamaindex/rebel_triples.json', 'w') as file:
    json.dump(distinct_triples, file)


##### llm & nebulargraph
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = 'yout openAi API key'

import logging
import sys

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  

from llama_index import (
    KnowledgeGraphIndex,
    LLMPredictor,
    ServiceContext,
    SimpleDirectoryReader,
)
from llama_index.storage.storage_context import StorageContext
from llama_index.graph_stores import NebulaGraphStore
from llama_index.llms import OpenAI

from IPython.display import Markdown, display

llm = OpenAI(temperature=0, model="text-davinci-002")
service_context = ServiceContext.from_defaults(llm=llm, chunk_size_limit=512)

# To set up NebulaGraph locally, begin by establishing a connection using its default credentials
# Install go --> sudo snap install go --classic
# Install nebula-console from https://github.com/vesoft-inc/nebula-console#from-source-code
# ./nebula-console -addr 127.0.0.1 -port 9669 -u root -p nebula
# CREATE SPACE llamaindex(vid_type=FIXED_STRING(256), partition_num=1, replica_factor=1);
# ADD HOSTS 127.0.0.1:9779;
# :sleep 10;
# USE llamaindex;
# CREATE TAG entity(name string);
# CREATE EDGE relationship(relationship string);
# CREATE TAG INDEX entity_index ON entity(name(256));

os.environ["NEBULA_USER"] = "root"
os.environ["NEBULA_PASSWORD"] = "nebula"
os.environ[
    "NEBULA_ADDRESS"
] = "127.0.0.1:9669"

space_name = "llamaindex"
edge_types, rel_prop_names = ["relationship"], [
    "relationship"
]
tags = ["entity"]

graph_store = NebulaGraphStore(
    space_name=space_name,
    edge_types=edge_types,
    rel_prop_names=rel_prop_names,
    tags=tags,
)
storage_context = StorageContext.from_defaults(graph_store=graph_store)

## Next, the data is loaded into the system using LlamaIndex’s SimpleDirectoryReader,
## which reads documents from a specified directory. A Knowledge Graph index, kg_index, is then constructed using these documents
from llama_index import SimpleDirectoryReader

print(os.getcwd())
reader = SimpleDirectoryReader(input_dir=os.getcwd() + "/kg/data/knowledge graphs/rebel_llamaindex/")
documents = reader.load_data()

kg_index = KnowledgeGraphIndex.from_documents(
    documents,
    storage_context=storage_context,
    max_triplets_per_chunk=5,
    service_context=service_context,
    space_name=space_name,
    edge_types=edge_types,
    rel_prop_names=rel_prop_names,
    tags=tags,
    include_embeddings=True,
)

from llama_index.query_engine import KnowledgeGraphQueryEngine

from llama_index.storage.storage_context import StorageContext
from llama_index.graph_stores import NebulaGraphStore

query_engine = KnowledgeGraphQueryEngine(
    storage_context=storage_context,
    service_context=service_context,
    llm=llm,
    verbose=True,
)


##### REBEL + LlamaIndex KnowledgeGraphIndex
space_name = "rebel_llamaindex"
edge_types, rel_prop_names = ["relationship"], [
    "relationship"
]  
tags = ["entity"]
graph_store = NebulaGraphStore(
    space_name=space_name,
    edge_types=edge_types,
    rel_prop_names=rel_prop_names,
    tags=tags,
)
storage_context = StorageContext.from_defaults(graph_store=graph_store)

from transformers import pipeline

triplet_extractor = pipeline('text2text-generation', model='Babelscape/rebel-large', tokenizer='Babelscape/rebel-large')
rebel_kg_index = KnowledgeGraphIndex.from_documents(
    documents,
    kg_triplet_extract_fn=extract_triplets,
    storage_context=storage_context,
    max_triplets_per_chunk=5,
    service_context=service_context,
    space_name=space_name,
    edge_types=edge_types,
    rel_prop_names=rel_prop_names,
    tags=tags,
    include_embeddings=True,
)
