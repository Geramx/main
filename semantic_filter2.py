from typing import List
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
import numpy as np

def quitar_redundancia_respetando_contenido(docs: List[Document], threshold: float = 0.9) -> List[Document]:
    # print("\n[Filtro] Iniciando filtro con preservación de contenido.")
    # print(f"[Filtro] Total de documentos recibidos: {len(docs)}")

    # for i, doc in enumerate(docs):
    #     # print(f"\n[Documento {i+1} original]:\n{doc.page_content}")

    # if not docs:
    #     return []

    embeddings = OpenAIEmbeddings()
    textos = [doc.page_content for doc in docs]
    vectores = embeddings.embed_documents(textos)

    únicos = [docs[0]]
    vectores_únicos = [vectores[0]]
    palabras_clave = set(textos[0].lower().split())

    descartados = []

    for i in range(1, len(vectores)):
        redundante = False
        palabras_actuales = set(textos[i].lower().split())

        for v_u, doc in zip(vectores_únicos, únicos):
            sim = np.dot(vectores[i], v_u) / (np.linalg.norm(vectores[i]) * np.linalg.norm(v_u))
            if sim > threshold:
                nuevas_palabras = palabras_actuales - palabras_clave
                if len(nuevas_palabras) < 3:
                    redundante = True
                    break

        if redundante:
            descartados.append(docs[i])
        else:
            únicos.append(docs[i])
            vectores_únicos.append(vectores[i])
            palabras_clave.update(palabras_actuales)

    # print(f"\n[Filtro] Documentos finales: {len(únicos)} (de {len(docs)} originales)")
    # print(f"[Filtro] Documentos eliminados por redundancia: {len(descartados)}")

    # for i, doc in enumerate(descartados):
    #     print(f"\n[Documento descartado {i+1}]:\n{doc.page_content}")

    return únicos
