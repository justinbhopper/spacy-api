from typing import List, Dict, Any, Optional
from enum import Enum
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from spacy import load as spacy_load
from spacy.tokens import Doc
from dependency_extraction import findSVs, findSVOs, findSVAOs

nlp = spacy_load("en_core_web_md")

# Set up the FastAPI app and define the endpoints
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])


# https://universaldependencies.org/u/pos/
POS_MAPPING = ["ADJ","ADP","ADV","AUX","CCONJ","DET","INTJ","NOUN","NUM","PART","PRON","PROPN","PUNCT","SCONJ","SYM","VERB","X"]


class ProcessRequestModel(BaseModel):
    articles: List[str]


class ProcessResponseModel(BaseModel):
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


def create_process_response(doc: Doc) -> Dict[str, Any]:
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


@app.post("/process", summary="Process batches of text", response_model=ProcessResponseModel)
def process_articles(query: ProcessRequestModel):
    """Process a batch of articles and return the entities predicted by the
    given model. Each record in the data should have a key "text".
    """
    response_body = []
    for doc in nlp.pipe(query.articles):
        response_body.append(create_process_response(doc))
    return {"result": response_body}


class SubjectVerbRequestModel(BaseModel):
    articles: List[str] = Field(..., title="Text to search")
    subjects: Optional[List[str]] = Field(None, title="Find matches only for the given subject lemmas")
    verbs: Optional[List[str]] = Field(None, title="Find matches only for the given verb lemmas")


class SubjectVerbResponseModel(BaseModel):
    class ArticleResult(BaseModel):
        subjectVerbs: List[List[str]]

    result: List[ArticleResult]


@app.post("/subject-verbs", summary="Finds subject-verb combinations", response_model=SubjectVerbResponseModel)
def find_subject_verbs(query: SubjectVerbRequestModel):
    result = []
    for doc in nlp.pipe(query.articles):
        subjectVerbs = []
        for pair in findSVs(doc, query.subjects, query.verbs):
            sub, verb = pair
            subjectVerbs.append([sub, verb])
        result.append({"subjectVerbs": subjectVerbs})
    return {"result": result}