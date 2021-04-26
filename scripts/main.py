from typing import List, Dict, Any
from enum import Enum
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import spacy
from spacy.tokens import Doc

nlp = spacy.load("en_core_web_md")


# https://universaldependencies.org/u/pos/
POS_MAPPING = ["ADJ","ADP","ADV","AUX","CCONJ","DET","INTJ","NOUN","NUM","PART","PRON","PROPN","PUNCT","SCONJ","SYM","VERB","X"]


class RequestModel(BaseModel):
    articles: List[str]


class ResponseModel(BaseModel):
    class Batch(BaseModel):
        class Entity(BaseModel):
            tokens: List[int]
            label: str

        ents: List[Entity] = []
        words: List[str]
        lemmas: List[str]
        offsets: List[int]
        pos: List[int]
        dep: List[str]
        lefts: Dict[int, List[int]]
        rights: Dict[int, List[int]]

    result: List[Batch]


def get_data(doc: Doc) -> Dict[str, Any]:
    ents = [
        {
            "tokens": [token.i for token in ent],
            "label": ent.label_
        }
        for ent in doc.ents
    ] 

    lefts = {}
    rights = {}
    for token in doc:
        tokenLefts = [left.i for left in token.lefts]
        if len(tokenLefts) > 0:
            lefts[token.i] = tokenLefts
        tokenRights = [right.i for right in token.rights]
        if len(tokenRights) > 0:
            rights[token.i] = tokenRights

    return {
        "ents": ents, 
        "words": [token.text for token in doc],
        "lemmas": [token.lemma_ for token in doc],
        "offsets": [token.idx for token in doc],
        "pos": [POS_MAPPING.index(token.pos_) for token in doc],
        "dep": [token.dep_ for token in doc],
        "lefts": lefts,
        "rights": rights
    }


# Set up the FastAPI app and define the endpoints
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])


@app.post("/process", summary="Process batches of text", response_model=ResponseModel)
def process_articles(query: RequestModel):
    """Process a batch of articles and return the entities predicted by the
    given model. Each record in the data should have a key "text".
    """
    response_body = []
    for doc in nlp.pipe(query.articles):
        response_body.append(get_data(doc))
    return {"result": response_body}