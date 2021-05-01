from typing import List, Dict, Any, Optional
from enum import Enum
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from lemminflect import isTagBaseForm, getLemma, getAllInflections, getAllInflectionsOOV
from spacy import load as spacy_load
from spacy.tokens import Doc
from dependency_extraction import findSVs, findSVOs, findSVAOs

nlp = spacy_load("en_core_web_md")

middleware = [Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=("GET", "POST"))]

# Set up the FastAPI app and define the endpoints
app = FastAPI(middleware=middleware)


# https://universaldependencies.org/u/pos/
POS_MAPPING = ["ADJ","ADP","ADV","AUX","CONJ","CCONJ","DET","EOL","INTJ","NOUN","NUM","PART","PRON","PROPN","PUNCT","SCONJ","SPACE","SYM","VERB","X"]


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


class PartOfSpeech(str, Enum):
    NOUN = 'NOUN' # Noun, singular
    NNS = 'NNS' # Noun, plural
    ADJ = 'ADJ' # Adjective
    VERB = 'VERB' # Verb
    VBD = 'VBD' # Verb, past tense


class LemmaResponseModel(BaseModel):
    lemma: str
    inflections: Dict[str, List[str]]


def merge_inflections(left, right):
    result = {}

    for pos in left:
        spellings = left[pos]
        result[pos] = spellings

    for pos in right:
        if (pos not in result):
            result[pos] = right[pos]

    return result


@app.get("/inflections", summary="Returns the lemmas of a given word", response_model=LemmaResponseModel)
def get_lemmas(word: str, pos: PartOfSpeech):
    word = word.lower()

    if (" " in word or "." in word):
        return JSONResponse (status_code = 200, content = {"message": "Input must contain only a single word without spaces or punctuation."})

    # Get the basic lemma version of the word first
    lemmas = getLemma(word, pos)
    if len(lemmas) > 0:
        lemma = getLemma(word, pos)[0]
    else:
        lemma = word

    inflections = merge_inflections(getAllInflections(lemma, upos=pos), getAllInflectionsOOV(lemma, upos=pos))
    
    return {"lemma": lemma, "inflections": inflections}
    