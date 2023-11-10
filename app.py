import os
from dotenv import load_dotenv
import chainlit as cl
import PyPDF2
from io import BytesIO
from langchain import HuggingFaceHub, PromptTemplate, LLMChain
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Initialiser le token API Hugging Face depuis les variables d'environnement
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

# ID du dépôt Hugging Face et initialisation du modèle
repo_id = "tiiuae/falcon-7b-instruct"
llm = HuggingFaceHub(
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    repo_id=repo_id,
    model_kwargs={"temperature": 0.7, "max_new_tokens": 500},
)

# Initialiser le séparateur de texte
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# Modèle de prompt pour les messages du chat
template = """Use the following pieces of context to answer the user's question...
... Begin!
----------------
{summaries}"""
messages = [
    SystemMessagePromptTemplate.from_template(template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = PromptTemplate(template=template, input_variables=["question"])
chain_type_kwargs = {"prompt": prompt}

@cl.on_chat_start
async def main():
    # Attente du téléchargement du fichier PDF
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="SVP Télécharger vos fichiers PDFs ici",
            accept=["application/pdf"],
        ).send()

    file = files[0]

    # Traitement du fichier PDF
    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    pdf_stream = BytesIO(file.content)
    pdf = PyPDF2.PdfReader(pdf_stream)
    pdf_text = ""
    for page in pdf.pages:
        pdf_text += page.extract_text()

    # Séparation du texte en chunks
    texts = text_splitter.split_text(pdf_text)
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]


    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    docsearch = await cl.make_async(Chroma.from_texts)(
        texts, embeddings, metadatas=metadatas
    )

    # Définir llm_chain comme une variable globale
    global llm_chain
    llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)

    cl.user_session.set("metadatas", metadatas)
    cl.user_session.set("texts", texts)

    msg.content = f"`{file.name}` Vous pouvez poser vos questions!"
    await msg.send()

    return llm_chain


@cl.on_message
async def main(res):
    # Imprimer la réponse complète pour identifier la clé correcte
    print(f"Complete response: {res}")

    # Récupérer la question depuis la réponse
    question = res.content

    # Ajouter du contexte ou reformuler la question si nécessaire
    context_question = f"Can you explain the concept of {question.lower()}?"

    # Appeler la chaîne de manière asynchrone
    res = await llm_chain.acall(context_question, callbacks=[cl.AsyncLangchainCallbackHandler()])

    # Vérifier si la clé "text" existe dans la réponse
    if "text" in res:
        answer = res["text"]
        sources = res.elements[0].content.strip() if res.elements else ""
        source_elements = []

        metadatas = cl.user_session.get("metadatas")
        all_sources = [m["source"] for m in metadatas]
        texts = cl.user_session.get("texts")

        if sources:
            found_sources = []

            for source in sources.split(","):
                source_name = source.strip().replace(".", "")
                try:
                    index = all_sources.index(source_name)
                except ValueError:
                    continue
                text = texts[index]
                found_sources.append(source_name)
                source_elements.append(cl.Text(content=text, name=source_name))

            if found_sources:
                answer += f"\nSources: {', '.join(found_sources)}"
            else:
                answer += "\nNo sources found"

        await cl.Message(content=answer, elements=source_elements).send()
    else:
        await cl.Message(content="Désolé, je n'ai pas pu traiter la question.").send()
