# pip install langchain openai unstructured chromadb tiktoken transformers flask flask_cors
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredPDFLoader
import nltk
import requests
import tempfile

from flask import Flask, request, render_template

from flask_cors import CORS

app = Flask(__name__)
app.static_folder = 'static'

CORS(app, supports_credentials=True)

nltk.download("punkt")

os.environ["OPENAI_API_KEY"] = "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"

app.config['chain'] = ''
app.config['text_splitter'] = ''
app.config['doc_search'] = ''
app.config['loaded_docs'] = ''

embeddings = OpenAIEmbeddings(openai_api_key = os.environ['OPENAI_API_KEY'])

def loadPDFFromURL(pdf_file_url):
  response = requests.get(pdf_file_url)
  with tempfile.NamedTemporaryFile(delete=False) as temp_file:
      temp_file.write(response.content)
      with open(temp_file.name, 'rb') as pdf_file:
          loader = UnstructuredPDFLoader(temp_file.name)
          loaded_docs = loader.load()
          temp_file.close()
          return loaded_docs

loaded_docs = loadPDFFromURL("https://firebasestorage.googleapis.com/v0/b/aire-consulting.appspot.com/o/Hotels_data.pdf?alt=media&token=257cc839-fadc-474f-99f4-1d0db6242230")
text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=0)
texts = text_splitter.split_documents(loaded_docs)
doc_search = Chroma.from_documents(texts,embeddings)

app.config['chain'] = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=doc_search)
    
@app.route("/")
def home():
    return render_template("index.html")

@app.route('/chat', methods=['POST', 'GET'])
def chat():
    try:
        question = request.args.get("msg")
        response = app.config['chain'].run(question)
        return str(response)
    except Exception as e:
        print("Error: ", e)
        return "Something went wrong !"

if __name__ == '__main__':
    app.run(port = '8000', debug=True)
