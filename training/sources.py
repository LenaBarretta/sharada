"""The label sets the base models are trained on, and the ones they are only measured on.

Everything here is a public dataset read through `datasets`. A source is a declaration: where the text
is, where the label is, what the options are called in plain words, and several ways of asking the
question. The builder turns one into `Example`s, and it varies three things on purpose:

* **the wording of the question**, so the model reads the question instead of memorising a task;
* **the order of the options**, which it cannot see anyway (`layout.py`) but which keeps the labels
  honest for any later variant that can;
* **which options are offered** — for a set of 77 intents most examples show a sampled handful, some
  show all of them, so both a short list and a long one are familiar.

Ordered options (`scale`) are never shuffled and never sampled: there, the order is the meaning.

Sources marked `holdout=True` are kept out of training entirely. They are the zero-shot measurement:
label sets the model has never been trained on, in a format it has.

A source that fails to load is skipped with a line saying so, never a crashed run — dataset ids on the
Hub move, and a three-hour training run should not die for one of them.
"""

from __future__ import annotations

import random
import zlib
from dataclasses import dataclass

from sharada import Example

MAX_OPTIONS = 24                    # the most that are ever shown at once
SUBSET_RANGE = (4, 12)              # how many are shown when a long list is sampled
FULL_LIST_SHARE = 0.25              # how often a long list is shown whole


@dataclass(frozen=True)
class Source:
    task: str                        # the name temperatures are fitted under
    dataset: str                     # id on the Hub
    text: str | tuple[str, ...]      # column, or columns joined into the text
    label: str | tuple[str, ...]     # column holding the answer; first one present wins
    questions: tuple[str, ...]       # phrasings; they take `{}` when `asks_about` is set
    config: str | None = None
    revision: str | None = None      # e.g. the Hub's auto-converted "refs/convert/parquet" branch
    kind: str = "choice"
    options: tuple[str, ...] | None = None    # plain-word option names, in label order
    rename: tuple[tuple[str, str], ...] = ()  # plain words for the dataset's own label names
    asks_about: str | None = None    # a second column that goes into the question
    train_split: str = "train"
    eval_split: str | None = "test"  # None: carve the last tenth out of the training split
    cap: int = 2500                  # at most this many examples per source
    single_label: bool = False       # the label column holds a list; keep the rows with one entry
    holdout: bool = False

    def as_dict(self) -> dict:
        return {"task": self.task, "dataset": self.dataset, "config": self.config,
                "revision": self.revision, "kind": self.kind, "holdout": self.holdout,
                "cap": self.cap}


# TREC's fifty fine-grained question types, which the dataset stores as codes like `ENTY:cremat`.
TREC_FINE = (
    ("ABBR:abb", "an abbreviation"), ("ABBR:exp", "what an abbreviation stands for"),
    ("ENTY:animal", "an animal"), ("ENTY:body", "a part of the body"), ("ENTY:color", "a colour"),
    ("ENTY:cremat", "a creative work"), ("ENTY:currency", "a currency"),
    ("ENTY:dismed", "a disease or a medicine"), ("ENTY:event", "an event"), ("ENTY:food", "a food"),
    ("ENTY:instru", "a musical instrument"), ("ENTY:lang", "a language"),
    ("ENTY:letter", "a letter of the alphabet"), ("ENTY:other", "some other kind of thing"),
    ("ENTY:plant", "a plant"), ("ENTY:product", "a product"), ("ENTY:religion", "a religion"),
    ("ENTY:sport", "a sport"), ("ENTY:substance", "a substance"), ("ENTY:symbol", "a symbol or a sign"),
    ("ENTY:techmeth", "a technique or a method"), ("ENTY:termeq", "an equivalent term"),
    ("ENTY:veh", "a vehicle"), ("ENTY:word", "a word with a special property"),
    ("DESC:def", "a definition"), ("DESC:desc", "a description"),
    ("DESC:manner", "the manner of doing something"), ("DESC:reason", "a reason"),
    ("HUM:gr", "a group or an organisation"), ("HUM:ind", "a person"),
    ("HUM:title", "the title of a person"), ("HUM:desc", "a description of a person"),
    ("LOC:city", "a city"), ("LOC:country", "a country"), ("LOC:mount", "a mountain"),
    ("LOC:other", "some other place"), ("LOC:state", "a state or a province"),
    ("NUM:code", "a code or a serial number"), ("NUM:count", "a count of things"),
    ("NUM:date", "a date"), ("NUM:dist", "a distance"), ("NUM:money", "an amount of money"),
    ("NUM:ord", "a position in an order"), ("NUM:other", "some other number"),
    ("NUM:period", "a length of time"), ("NUM:perc", "a percentage"), ("NUM:speed", "a speed"),
    ("NUM:temp", "a temperature"), ("NUM:volsize", "a size or a volume"), ("NUM:weight", "a weight"),
)


# arXiv files papers under codes; the model reads its options, so they are spelled out.
ARXIV = (
    ("math.AC", "commutative algebra"), ("math.GR", "group theory"), ("math.ST", "theoretical statistics"),
    ("cs.AI", "artificial intelligence"), ("cs.CV", "computer vision"), ("cs.NE", "neural networks"),
    ("cs.SY", "systems and control"), ("cs.CE", "computational science and engineering"),
    ("cs.PL", "programming languages"), ("cs.IT", "information theory"),
    ("cs.DS", "data structures and algorithms"),
)


# ── what it is trained on ────────────────────────────────────────────────────────────────────────

SOURCES: tuple[Source, ...] = (
    # intents: many options, short texts
    Source("banking-intent", "mteb/banking77", "text", ("label_text", "label"),
           ("Which banking request is this?",
            "What is this customer asking about?",
            "Pick the intent of this message to a bank."),
           cap=4000),
    Source("clinc-intent", "clinc/clinc_oos", "text", "intent",
           ("Which assistant request is this?",
            "What is the user asking the assistant to do?",
            "Pick the intent."),
           config="plus", cap=4000, eval_split="validation"),
    Source("massive-intent", "mteb/amazon_massive_intent", "text", ("label_text", "label"),
           ("Which assistant request is this?",
            "What does the user want?",
            "Pick the intent of this command."),
           config="en", cap=4000, eval_split="validation"),

    # topics
    Source("news-section", "fancyzhx/ag_news", "text", "label",
           ("Which section does this news item belong to?",
            "Which desk would run this story?",
            "What is this article about?"),
           options=("world news", "sports", "business", "science and technology"), cap=3000),
    Source("entity-type", "fancyzhx/dbpedia_14", "content", "label",
           ("What kind of thing is this article about?",
            "Which category does the subject of this text belong to?"),
           cap=3000),
    Source("forum-topic", "community-datasets/yahoo_answers_topics", ("question_title", "question_content"), "topic",
           ("Which topic was this question posted under?",
            "Which board does this question belong on?"),
           cap=3000),
    Source("newsgroup", "SetFit/20_newsgroups", "text", ("label_text", "label"),
           ("Which newsgroup was this posted to?",
            "Which group does this message belong in?"),
           cap=2500),
    Source("question-type", "CogComp/trec", "text", "coarse_label",
           ("What is this question asking for?",
            "What kind of answer would this question need?"),
           options=("an abbreviation", "a description", "an entity", "a person", "a place", "a number"),
           revision="refs/convert/parquet", cap=2000),
    Source("question-type-fine", "CogComp/trec", "text", "fine_label",
           ("Precisely what is this question asking for?",
            "What exactly would the answer to this be?"),
           revision="refs/convert/parquet", rename=TREC_FINE, cap=2000),

    # feeling, on a scale
    Source("review-stars", "Yelp/yelp_review_full", "text", "label",
           ("How many stars did this review give?",
            "How happy is this customer?",
            "Rate the tone of this review."),
           kind="scale", options=("1 star", "2 stars", "3 stars", "4 stars", "5 stars"), cap=4000),
    Source("app-stars", "sealuzh/app_reviews", "review", "star",
           ("How many stars did this review give?",
            "How satisfied is this user with the app?"),
           kind="scale", options=("1 star", "2 stars", "3 stars", "4 stars", "5 stars"),
           cap=2500, eval_split=None),
    Source("sentence-tone", "SetFit/sst5", "text", "label",
           ("How positive is this sentence?",
            "Where does this land between praise and a panning?"),
           kind="scale",
           options=("very negative", "negative", "neutral", "positive", "very positive"),
           cap=2500, eval_split="test"),
    Source("tweet-sentiment", "cardiffnlp/tweet_eval", "text", "label",
           ("How positive is this tweet?",
            "What is the sentiment here?"),
           config="sentiment", kind="scale", options=("negative", "neutral", "positive"), cap=2500),
    Source("movie-verdict", "stanfordnlp/imdb", "text", "label",
           ("Did this reviewer like the film?",
            "Is this review positive or negative?"),
           kind="scale", options=("negative", "positive"), cap=2500),
    Source("short-verdict", "cornell-movie-review-data/rotten_tomatoes", "text", "label",
           ("Is this one-line review positive or negative?",
            "Did this critic like it?"),
           kind="scale", options=("negative", "positive"), cap=2000, eval_split="validation"),
    Source("product-tone", "fancyzhx/amazon_polarity", "content", "label",
           ("Is this product review positive or negative?",
            "Was this customer happy with what they bought?"),
           kind="scale", options=("negative", "positive"), cap=2500),

    # emotion
    Source("emotion", "dair-ai/emotion", "text", "label",
           ("What is the writer feeling?",
            "Which emotion does this message carry?"),
           options=("sadness", "joy", "love", "anger", "fear", "surprise"), cap=2500),
    Source("tweet-emotion", "cardiffnlp/tweet_eval", "text", "label",
           ("What is this tweet feeling?",
            "Which emotion is behind this?"),
           config="emotion", options=("anger", "joy", "optimism", "sadness"), cap=2000),
    Source("fine-emotion", "google-research-datasets/go_emotions", "text", "labels",
           ("Which feeling does this comment carry?",
            "What is the commenter feeling?"),
           config="simplified", single_label=True, cap=3000),

    # yes or no
    Source("spam", "ucirvine/sms_spam", "sms", "label",
           ("Is this text message spam?",
            "Would you move this message to junk?"),
           kind="binary", options=("no", "yes"), cap=2000, eval_split=None),
    Source("offensive", "cardiffnlp/tweet_eval", "text", "label",
           ("Is this message offensive?",
            "Would this get reported?"),
           config="offensive", kind="binary", options=("no", "yes"), cap=2500),
    Source("hateful", "cardiffnlp/tweet_eval", "text", "label",
           ("Is this message hateful towards a group of people?",
            "Does this attack a group?"),
           config="hate", kind="binary", options=("no", "yes"), cap=2500,
           # Its own test split was built to be a different problem from its training split, and
           # everything scores around chance on it; the tenth carved off the training data measures
           # what the model actually learned.
           eval_split=None),
    Source("irony", "cardiffnlp/tweet_eval", "text", "label",
           ("Is this meant ironically?",
            "Does the writer mean the opposite of what they wrote?"),
           config="irony", kind="binary", options=("no", "yes"), cap=2000),
    Source("toxic-comment", "SetFit/toxic_conversations", "text", "label",
           ("Is this comment toxic?",
            "Would a moderator take this down?"),
           kind="binary", options=("no", "yes"), cap=2500),
    Source("grammatical", "nyu-mll/glue", "sentence", "label",
           ("Is this sentence grammatical English?",
            "Would a native speaker write it this way?"),
           config="cola", kind="binary", options=("no", "yes"), cap=2500,
           eval_split="validation"),

    # the question carries the second half, which is what teaches it to read the question
    Source("entailment", "nyu-mll/glue", "premise", "label",
           ("Does it follow from this that {}?",
            "If the text is true, is it also true that {}?",
            "Given this, how likely is it that {}?"),
           config="mnli", kind="scale", options=("yes", "maybe", "no"),
           asks_about="hypothesis", cap=4000, train_split="train",
           eval_split="validation_matched"),
    Source("entailment-short", "stanfordnlp/snli", "premise", "label",
           ("Does this mean that {}?",
            "If this is the scene, is it true that {}?"),
           kind="scale", options=("yes", "maybe", "no"), asks_about="hypothesis", cap=3000),
    Source("follows", "nyu-mll/glue", "sentence1", "label",
           ("Does this support the claim that {}?",
            "Is the claim '{}' true according to this text?"),
           config="rte", kind="binary", options=("yes", "no"), asks_about="sentence2",
           cap=2000, eval_split="validation"),
    Source("answers-question", "nyu-mll/glue", "sentence", "label",
           ("Does this sentence answer the question '{}'?",
            "Is the answer to '{}' in this sentence?"),
           config="qnli", kind="binary", options=("yes", "no"), asks_about="question",
           cap=3000, eval_split="validation"),
    Source("same-meaning", "nyu-mll/glue", "sentence1", "label",
           ("Does this say the same thing as '{}'?",
            "Are this and '{}' the same statement in different words?"),
           config="mrpc", kind="binary", options=("no", "yes"), asks_about="sentence2",
           cap=2000, eval_split="validation"),
    Source("same-question", "nyu-mll/glue", "question1", "label",
           ("Is this the same question as '{}'?",
            "Would the answer to '{}' answer this too?"),
           config="qqp", kind="binary", options=("no", "yes"), asks_about="question2",
           cap=3000, eval_split="validation"),
    Source("paraphrase", "google-research-datasets/paws", "sentence1", "label",
           ("Does this mean the same as '{}'?",
            "Same statement as '{}', or a different one?"),
           config="labeled_final", kind="binary", options=("no", "yes"), asks_about="sentence2",
           cap=2500),

    # ── held out of training, measured only ──────────────────────────────────────────────────
    Source("massive-scenario", "mteb/amazon_massive_scenario", "text", ("label_text", "label"),
           ("Which part of the assistant does this belong to?",
            "What area is this command about?"),
           config="en", cap=2000, eval_split="validation", holdout=True),
    Source("claim-veracity", "ImperialCollegeLondon/health_fact", "main_text", "label",
           ("How true is the claim this text is checking?",
            "What verdict does this fact-check reach?"),
           options=("false", "partly true", "true", "unproven"),
           revision="refs/convert/parquet", cap=1500, holdout=True),
    Source("arxiv-category", "ccdv/arxiv-classification", "text", "label",
           ("Which arXiv category was this paper filed under?",
            "Which part of the literature does this paper belong to?",
            "What is this paper about?"),
           config="no_ref", rename=ARXIV, cap=1500, eval_split="test", holdout=True),
    Source("medical-pair", "curaihealth/medical_questions_pairs", "question_1", "label",
           ("Is this the same medical question as '{}'?",
            "Would one answer serve both this and '{}'?"),
           kind="binary", options=("no", "yes"), asks_about="question_2",
           cap=1500, eval_split=None, holdout=True),
    Source("subjective", "SetFit/subj", "text", ("label_text", "label"),
           ("Is this sentence an opinion or a statement of fact?",
            "Is the writer describing or judging?"),
           cap=1500, holdout=True),
    Source("poem-tone", "google-research-datasets/poem_sentiment", "verse_text", "label",
           ("What is the feeling of this line of verse?",
            "How does this line read?"),
           kind="scale", options=("negative", "positive", "neither", "mixed"),
           cap=800, eval_split="validation", holdout=True),
)


# ── turning a source into examples ───────────────────────────────────────────────────────────────

class Unusable(Exception):
    """This source cannot be read as declared; the run goes on without it."""


def plain_words(name: str) -> str:
    """`EducationalInstitution`, `card_arrival` -> `educational institution`, `card arrival`."""
    spaced = []
    for part in str(name).replace("_", " ").replace("-", " ").split():
        out = part[:1]
        for previous, letter in zip(part, part[1:]):
            out += (" " if letter.isupper() and not previous.isupper() else "") + letter
        spaced.append(out)
    return " ".join(" ".join(spaced).split()).lower()


def tidy(value) -> str:
    return " ".join(str(value or "").split())


def _column(row_or_features, names) -> str:
    for name in (names,) if isinstance(names, str) else names:
        if name in row_or_features:
            return name
    raise Unusable(f"none of {names} is a column; it has {list(row_or_features)[:8]}")


def _answers(rows, column: str, declared, rename=()) -> tuple[list[str], dict]:
    """The option names, and the map from what is in the column to an option index."""
    feature = rows.features[column]
    names = list(getattr(feature, "names", None) or getattr(getattr(feature, "feature", None), "names", None) or [])
    if names:
        keys = list(range(len(names)))
    else:
        values = sorted(rows.unique(column))
        if len(values) > 200:
            raise Unusable(f"{len(values)} distinct labels in {column!r}")
        if not declared and not all(isinstance(v, str) for v in values):
            raise Unusable(f"{column!r} holds plain numbers and the source declares no option names")
        if declared and any(isinstance(v, str) for v in values):
            raise Unusable(f"{column!r} holds strings, so declared option names cannot be matched"
                           " to them by order — point `label` at the numeric column")
        keys, names = values, [plain_words(v) for v in values]
    spelled = dict(rename)
    options = list(declared) if declared else [spelled.get(str(n), plain_words(n)) for n in names]
    if len(options) != len(names):
        raise Unusable(f"{len(options)} option names declared for {len(names)} labels")
    return options, {key: i for i, key in enumerate(keys)}


def _rows(source: Source, which: str, seed: int):
    from datasets import load_dataset

    split = source.train_split if which == "train" else (source.eval_split or source.train_split)
    rows = load_dataset(source.dataset, source.config, split=split, revision=source.revision)
    if source.eval_split is None:                       # one split only: carve off the last tenth
        rows = rows.shuffle(seed=7)
        cut = max(1, len(rows) // 10)
        rows = rows.select(range(cut, len(rows))) if which == "train" else rows.select(range(cut))
    return rows.shuffle(seed=seed)


def examples(source: Source, which: str = "train", seed: int = 0, cap: int | None = None,
             full_options: bool = False) -> list[Example]:
    """The examples this source contributes, or `Unusable` if it cannot be read as declared.

    `full_options` offers every label at once instead of a sampled handful. That is what a measurement
    should do: a user with 77 intents sends all 77, and accuracy over six of them sampled at random is
    a different, easier question.
    """
    rows = _rows(source, which, seed)
    label = _column(rows.features, source.label)
    text_columns = (source.text,) if isinstance(source.text, str) else source.text
    for column in text_columns:
        _column(rows.features, column)
    if source.asks_about:
        _column(rows.features, source.asks_about)

    rows = rows.filter(lambda row: row[label] not in (-1, None, "", "-1"))
    if source.single_label:
        rows = rows.filter(lambda row: len(row[label]) == 1)
    options, index = _answers(rows, label, source.options, source.rename)

    limit = min(len(rows), cap if cap is not None else source.cap)
    rows = rows.select(range(limit))
    ordered = source.kind == "scale"
    rng = random.Random(zlib.crc32(f"{source.task}/{which}/{seed}".encode()))

    out = []
    for row in rows:
        text = "\n".join(tidy(row[column]) for column in text_columns if tidy(row[column]))
        if not text:
            continue
        answer = row[label][0] if source.single_label else row[label]
        truth = index.get(answer)
        if truth is None:
            continue
        question = rng.choice(source.questions)
        if source.asks_about:
            about = tidy(row[source.asks_about])
            if not about:
                continue
            question = question.format(about)
        shown, truth = _offer(options, truth, ordered, rng, full_options)
        out.append(Example(text=text, question=question, options=shown, kind=source.kind,
                           task=source.task, label=truth))
    if not out:
        raise Unusable("no usable rows")
    return out


def _offer(options: list[str], truth: int, ordered: bool, rng: random.Random,
           full: bool = False) -> tuple[list[str], int]:
    """Which options this example shows, and where the right one ended up."""
    if ordered or full:
        return list(options), truth                      # a scale is shown whole, in its own order
    shown = list(range(len(options)))
    if len(shown) > MAX_OPTIONS or (len(shown) > SUBSET_RANGE[1] and rng.random() > FULL_LIST_SHARE):
        keep = min(len(shown), rng.randint(*SUBSET_RANGE))
        others = [i for i in shown if i != truth]
        shown = [truth] + rng.sample(others, keep - 1)
    rng.shuffle(shown)
    return [options[i] for i in shown], shown.index(truth)


def build(which: str = "train", seed: int = 0, cap_scale: float = 1.0,
          include_holdout: bool = False, variants: int = 1, full_options: bool = False,
          sources=SOURCES, log=print) -> list[Example]:
    """Every source that loads, in one shuffled list. Sources that do not load are skipped out loud.

    `variants` draws each source more than once. A second pass over the same rows asks them through a
    different wording of the question and offers a different subset of the options, and where the
    source has more rows than its cap, it reaches different rows as well. For the small label sets
    that is the only way to see more than a few thousand examples of them, and the wording is the
    thing the model is supposed to be reading.
    """
    collected: list[Example] = []
    skipped = []
    for source in sources:
        if source.holdout and not include_holdout:
            continue
        cap = max(20, int(source.cap * cap_scale))
        try:
            got = []
            for pass_ in range(max(1, variants)):
                got += examples(source, which, seed + 97 * pass_, cap, full_options)
        except Unusable as problem:
            skipped.append((source, str(problem)))
            log(f"  skipped {source.task} ({source.dataset}): {problem}")
            continue
        except Exception as problem:                     # a moved id, a gated dataset, no network
            skipped.append((source, f"{type(problem).__name__}: {problem}"))
            log(f"  skipped {source.task} ({source.dataset}): {type(problem).__name__}: {problem}")
            continue
        collected += got
        log(f"  {source.task:18s} {len(got):6d} {which:5s} {len(got[0].options):3d} options"
            f"{'  (held out of training)' if source.holdout else ''}")
    random.Random(seed + 1).shuffle(collected)
    log(f"{len(collected)} {which} examples from {len({e.task for e in collected})} label sets,"
        f" {len(skipped)} sources skipped")
    return collected


def holdout_sources() -> tuple[Source, ...]:
    return tuple(s for s in SOURCES if s.holdout)


def catalogue() -> list[dict]:
    return [s.as_dict() for s in SOURCES]
