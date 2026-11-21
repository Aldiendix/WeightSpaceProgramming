"""Generated reasoning puzzles of the kinds BIG-Bench Hard tests (practice:bbhsynth): every item is written by a
program from its own word lists and solved by the same program, so its answer is exact and it shares no question with
the benchmark (the generated set is checked against data/bench_fingerprints.txt.gz). Only kinds a program can pose and
solve are generated; kinds that rest on world knowledge or taste (sports, movies, puns, sarcasm, translation errors,
causal judgement, pronoun reference) and dyck_languages (an answer of brackets, which cannot be given inside
\\boxed{}) are not. Items: task "exact", the answer an option letter "(A)", a word ("True", "Yes", "valid"), a number
or a word list, graded by grade._exact.

make_bbhsynth() writes data/practice/bbhsynth.jsonl.gz once; the set is frozen like the other practice sets.
"""
import datetime
import itertools
import random
import re

from .common import read_jsonl, write_jsonl

LETTERS = "ABCDEFGHIJKLMNOPQR"
NUMBER_WORDS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
                "seventeen eighteen nineteen twenty").split()
ORDINALS = ("first second third fourth fifth sixth seventh").split()
PEOPLE = ("Alice", "Bob", "Claire", "Dave", "Eve", "Fred", "Gertrude")


def _options(choices):
    return "\n".join(f"({LETTERS[i]}) {c}" for i, c in enumerate(choices))


def _choice(question, choices, right):
    return f"{question}\nOptions:\n{_options(choices)}", f"({LETTERS[choices.index(right)]})"


def _a(word):
    return ("an " if word[0] in "aeiou" else "a ") + word


def _list(words):
    if len(words) == 1:
        return words[0]
    return ", ".join(words[:-1]) + (", and " if len(words) > 2 else " and ") + words[-1]


# --- boolean expressions, arithmetic ------------------------------------------------------------------------------

def boolean_expressions(rng):
    def expr(depth):
        if depth == 0 or rng.random() < 0.3:
            s = rng.choice(["True", "False"])
        else:
            s = f"{expr(depth - 1)} {rng.choice(['and', 'or'])} {expr(depth - 1)}"
            if rng.random() < 0.5:
                s = f"( {s} )"
        return "not " * rng.choice([0, 0, 1, 1, 2]) + s
    while True:
        s = expr(2)
        if 3 <= sum(s.count(v) for v in ("True", "False")) <= 5:
            return f"{s} is", str(eval(s))  # noqa: S307 - our own expression of True/False/and/or/not/()


def multistep_arithmetic(rng):
    def group():
        terms = [str(rng.randint(-9, 9)) for _ in range(4)]
        return "(" + "".join(f" {rng.choice('+-*')} {t}" if i else t for i, t in enumerate(terms)) + ")"
    s = f"({group()} {rng.choice('+-*')} {group()})"
    return f"{s} =", str(eval(s))  # noqa: S307 - integers and + - * only


# --- counting -------------------------------------------------------------------------------------------------------

COUNT_KINDS = {
    "musical instruments": "flute piano trombone violin accordion clarinet drum trumpet cello harp saxophone banjo "
                           "tuba oboe xylophone",
    "fruits": "apple banana peach plum orange grape nectarine blackberry raspberry strawberry mango lemon pear apricot "
              "kiwi",
    "vegetables": "carrot potato onion cabbage yam garlic turnip zucchini eggplant radish leek pumpkin "
                  "cauliflower lettuce",
    "animals": "cat dog goat rabbit pig cow donkey mouse snake frog duck chicken horse sheep fish",
    "objects": "chair table bed lamp couch stove oven microwave fridge toaster car",
}
PLURAL = {"peach": "peaches", "mango": "mangoes", "potato": "potatoes", "radish": "radishes", "mouse": "mice",
          "fish": "fish", "sheep": "sheep", "couch": "couches", "strawberry": "strawberries",
          "blackberry": "blackberries", "raspberry": "raspberries", "cabbage": "cabbages",
          "lettuce": "heads of lettuce",
          "cauliflower": "heads of cauliflower", "garlic": "heads of garlic"}


def _count_phrase(word, k):
    if k == 1:
        return "a head of " + word if word in ("lettuce", "cauliflower", "garlic") else _a(word)
    return f"{NUMBER_WORDS[k]} {PLURAL.get(word, word + 's')}"


def object_counting(rng):
    kind = rng.choice(list(COUNT_KINDS))
    words = rng.sample(COUNT_KINDS[kind].split(), rng.randint(3, 9))
    items = [(w, rng.choice([1, 1, 1, 2, 3, 4]), True) for w in words]
    others = [k for k in COUNT_KINDS if k != kind and not {k, kind} <= {"objects", "musical instruments"}]
    for w in rng.sample(COUNT_KINDS[rng.choice(others)].split(), rng.randint(0, 3)):
        items.insert(rng.randrange(len(items) + 1), (w, rng.choice([1, 2, 3]), False))
    question = f"I have {_list([_count_phrase(w, k) for w, k, _ in items])}. How many {kind} do I have?"
    return question, str(sum(k for _, k, own in items if own))


# --- word sorting ---------------------------------------------------------------------------------------------------

def _vocabulary():
    from .data import PRACTICE_DIR  # noqa: PLC0415
    words = set()
    for r in read_jsonl(PRACTICE_DIR / "gsm8k_train.jsonl.gz"):
        words |= {w for w in re.findall(r"\b[a-z]+\b", r["problem"]) if 3 <= len(w) <= 12}
    return sorted(words)


def word_sorting(rng, vocab):
    words = rng.sample(vocab, rng.randint(3, 18))
    return f"Sort the following words alphabetically: List: {' '.join(words)}", " ".join(sorted(words))


# --- navigation, truth-tellers --------------------------------------------------------------------------------------

def navigate(rng):
    want = rng.random() < 0.45
    if rng.random() < 0.5:  # always facing forward: steps in four directions
        dirs = {"forward": (0, 1), "backward": (0, -1), "left": (-1, 0), "right": (1, 0)}
        while True:
            moves = [(rng.choice(list(dirs)), rng.randint(1, 10)) for _ in range(rng.randint(2, 6))]
            if want:
                x = sum(dirs[d][0] * n for d, n in moves)
                y = sum(dirs[d][1] * n for d, n in moves)
                moves += [("left" if x > 0 else "right", abs(x))] if x else []
                moves += [("backward" if y > 0 else "forward", abs(y))] if y else []
                rng.shuffle(moves)
            pos = [sum(dirs[d][i] * n for d, n in moves) for i in (0, 1)]
            if 2 <= len(moves) <= 9 and all(n <= 10 for _, n in moves) and (pos == [0, 0]) == want:
                break
        steps = " ".join(f"Take {n} step{'s' * (n > 1)} {d}." for d, n in moves)
        text = f"Always face forward. {steps}"
    else:  # turning: steps are taken in the direction faced
        while True:
            acts, h, x, y = [], 0, 0, 0
            for _ in range(rng.randint(3, 8)):
                if rng.random() < 0.35:
                    t = rng.choice(["Turn left.", "Turn right.", "Turn around."])
                    h = (h + {"Turn left.": 3, "Turn right.": 1, "Turn around.": 2}[t]) % 4
                    acts.append(t)
                else:
                    n = rng.randint(1, 10)
                    x, y = x + n * (0, 1, 0, -1)[h], y + n * (1, 0, -1, 0)[h]
                    acts.append(f"Take {n} step{'s' * (n > 1)}.")
            if want:  # walk back: face the way that undoes y, then x (headings 0 north, 1 east, 2 south, 3 west)
                for target, n in ((2 if y > 0 else 0, abs(y)), (3 if x > 0 else 1, abs(x))):
                    if n:
                        turn = (target - h) % 4
                        if turn:
                            acts.append({1: "Turn right.", 2: "Turn around.", 3: "Turn left."}[turn])
                            h = target
                        x, y = x + n * (0, 1, 0, -1)[h], y + n * (1, 0, -1, 0)[h]
                        acts.append(f"Take {n} step{'s' * (n > 1)}.")
            if len(acts) <= 12 and ((x, y) == (0, 0)) == want and "step" in " ".join(acts):
                break
        text = " ".join(acts)
    return (f"If you follow these instructions, do you return to the starting point? {text}\nOptions:\n- Yes\n- No",
            "Yes" if want else "No")


NAMES = ("Amberly Bernita Caden Delfina Emmett Fidel Gwendolyn Harlan Ione Jasper Kendra Leopold Marisol Nadine "
         "Osvaldo Perla Quincy Rosalind Silas Tamsin Ulysses Valentina Wendell Ximena Yusuf Zelda Anselm Brielle "
         "Cosimo Dagny Eamon Fenna Gideon Hollis Imogen Joaquin").split()


def web_of_lies(rng):
    people = rng.sample(NAMES, 5)
    truth = rng.random() < 0.5
    parts = [f"{people[0]} {'tells the truth' if truth else 'lies'}."]
    for prev, p in zip(people, people[1:]):
        says_truth = rng.random() < 0.5
        parts.append(f"{p} says {prev} {'tells the truth' if says_truth else 'lies'}.")
        truth = truth == says_truth
    return f"Question: {' '.join(parts)} Does {people[-1]} tell the truth?", "Yes" if truth else "No"


# --- tracking swaps -------------------------------------------------------------------------------------------------

SWAP_SCENES = [
    ("{who} are playing a game. At the start of the game, they are each holding a ball: {start}.\nAs the game "
     "progresses, pairs of players trade balls. {swaps} At the end of the game, {q} has the",
     "{p} has {x}", "{a} and {b} swap balls",
     [c + " ball" for c in "red orange yellow green blue purple pink white black brown".split()], True),
    ("{who} are friends and avid readers who occasionally trade books. At the start of the semester, they each buy one "
     "new book: {start}.\nAs the semester proceeds, they start trading around the new books. {swaps} At the end of the "
     "semester, {q} has",
     "{p} gets {x}", "{a} and {b} swap books",
     ["Middlemarch", "Dracula", "Emma", "Walden", "Beloved", "Persuasion", "Rebecca", "Ivanhoe", "The Iliad",
      "War and Peace", "Don Quixote", "Jane Eyre"], False),
    ("{who} are dancers at a square dance. At the start of a song, they each have a partner: {start}.\nThroughout the "
     "song, the dancers often trade partners. {swaps} At the end of the dance, {q} is dancing with",
     "{p} is dancing with {x}", "{a} and {b} switch partners",
     ["Mirela", "Tobias", "Ingrid", "Ramon", "Svetla", "Hugo", "Annika", "Matteo", "Odile", "Pablo"], False),
    ("{who} are on the same team in a soccer match. At the start of the match, they are each assigned to a position: "
     "{start}.\nAs the game progresses, pairs of players occasionally swap positions. {swaps} At the end of the match, "
     "{q} is playing",
     "{p} is playing {x}", "{a} and {b} trade positions",
     ["goalkeeper", "left winger", "right winger", "striker", "center midfielder", "left midfielder", "sweeper",
      "cheerleader", "fullback", "benchwarmer"], False),
    ("{who} are holding a white elephant gift exchange. At the start of the event, they are each holding a present of "
     "a different color: {start}.\nAs the event progresses, pairs of people swap gifts. {swaps} At the end of the "
     "event, {q} has the",
     "{p} has {x}", "{a} and {b} swap their gifts",
     [c + " present" for c in "red orange yellow green blue purple pink white black brown".split()], True),
]


def tracking_shuffled_objects(rng, n):
    text, has, swap, pool, article = rng.choice(SWAP_SCENES)
    people, things = list(PEOPLE[:n]), rng.sample(pool, n)
    held = dict(zip(people, things))
    swaps = []
    for _ in range(n):
        a, b = rng.sample(people, 2)
        held[a], held[b] = held[b], held[a]
        swaps.append(swap.format(a=a, b=b))
    words = ["First"] + ["Then"] * (n - 2) + ["Finally"]
    q = rng.choice(people)
    question = text.format(who=_list(people), q=q,
                           start=_list([has.format(p=p, x=_a(x) if article else x) for p, x in zip(people, things)]),
                           swaps=" ".join(f"{w}, {s}." for w, s in zip(words, swaps)))
    return _choice(question, things, held[q])


# --- ordering by clues ----------------------------------------------------------------------------------------------

def _rank_phrases(lo, hi, lo2, hi2, the="the "):
    """Position i of n (0 = the lo end) -> phrase."""
    def phrase(i, n):
        if i == 0:
            return lo
        if i == n - 1:
            return hi
        if i < n - 1 - i or (i == n - 1 - i and i % 2):
            return f"{the}{ORDINALS[i]}{lo2}"
        return f"{the}{ORDINALS[n - 1 - i]}{hi2}"
    return phrase


ORDER_SCENES = [  # (setting, items, item -> mention, verb, i<j relation, j<i relation, rank phrase)
    ("On a branch, there are {n} birds: {items}.",
     "quail owl raven falcon robin crow hummingbird cardinal hawk sparrow magpie heron".split(),
     lambda x: f"the {x}", "is", "to the left of", "to the right of",
     _rank_phrases("the leftmost", "the rightmost", " from the left", " from the right")),
    ("On a shelf, there are {n} books: {items}.",
     [c + " book" for c in "red blue green yellow orange purple white black grey brown".split()],
     lambda x: f"the {x}", "is", "to the left of", "to the right of",
     _rank_phrases("the leftmost", "the rightmost", " from the left", " from the right")),
    ("In an antique car show, there are {n} vehicles: {items}.",
     "sedan truck limousine motorcycle station-wagon tractor bus convertible minivan hatchback coupe".split(),
     lambda x: f"the {x}", "is", "older than", "newer than",
     _rank_phrases("the oldest", "the newest", "-oldest", "-newest")),
    ("In a golf tournament, there were {n} golfers: {items}.",
     "Dana Joe Mel Rob Ana Eli Ada Mya Ivy Amy Eve Ken Bo Lu".split(),
     lambda x: x, "", "finished above", "finished below",
     _rank_phrases("first", "last", "", "-to-last", the="")),
    ("A fruit stand sells {n} fruits: {items}.",
     "apples kiwis plums mangoes watermelons pears peaches oranges loquats cantaloupes".split(),
     lambda x: f"the {x}", "are", "less expensive than", "more expensive than",
     _rank_phrases("the cheapest", "the most expensive", "-cheapest", "-most expensive")),
]


def logical_deduction(rng, n):
    setting, pool, mention, verb, before, after, rank = rng.choice(ORDER_SCENES)
    items = rng.sample(pool, n)
    truth = rng.sample(items, n)  # truth[i] holds position i
    pos = {x: i for i, x in enumerate(truth)}
    clues = [("rank", x, None) for x in items] + [("rel", x, y) for x in items for y in items if x != y]
    rng.shuffle(clues)
    perms = [dict(zip(items, p)) for p in itertools.permutations(range(n))]  # item -> position, every order

    def holds(p, c):
        return p[c[1]] == pos[c[1]] if c[0] == "rank" else (p[c[1]] < p[c[2]]) == (pos[c[1]] < pos[c[2]])
    cands, chosen = perms, []
    for c in clues:
        if len(cands) == 1:
            break
        kept = [p for p in cands if holds(p, c)]
        if len(kept) < len(cands):
            cands, chosen = kept, chosen + [c]
    for c in list(chosen):  # drop clues the others imply
        rest = [d for d in chosen if d is not c]
        if sum(all(holds(p, d) for d in rest) for p in perms) == 1:
            chosen = rest

    def sentence(c):
        x = mention(c[1])
        s = f"{x[0].upper()}{x[1:]}{' ' + verb if verb else ''}"
        if c[0] == "rank":
            return f"{s} {'finished ' if not verb else ''}{rank(pos[c[1]], n)}."
        rel = before if pos[c[1]] < pos[c[2]] else after
        return f"{s} {rel} {mention(c[2])}."
    listing = _list([x if not verb else (x if verb == "are" else _a(x)) for x in items])
    head = (f"The following paragraphs each describe a set of {NUMBER_WORDS[n]} objects arranged in a fixed order. The "
            "statements are logically consistent within each paragraph. ")
    text = head + setting.format(n=NUMBER_WORDS[n], items=listing) + " " + " ".join(sentence(c) for c in chosen)
    stated = {pos[c[1]] for c in chosen if c[0] == "rank"}
    k = rng.choice([i for i in range(n) if i not in stated] or list(range(n)))  # not a position a clue states
    opts = []
    for x in items:
        m = mention(x)
        opts.append(f"{m[0].upper()}{m[1:]}{' ' + verb if verb else ''} {'finished ' if not verb else ''}{rank(k, n)}")
    return _choice(text, opts, opts[items.index(truth[k])])


# --- dates and times ------------------------------------------------------------------------------------------------

def _mdy(d):
    return d.strftime("%m/%d/%Y")


def date_understanding(rng):
    while True:
        today = datetime.date(rng.randint(1900, 2030), 1, 1) + datetime.timedelta(days=rng.randint(0, 364))
        if today.day <= 28:
            break
    form = rng.randrange(6)
    if form == 0:
        lead = f"Today is {_mdy(today)}."
    elif form == 1:
        lead = f"Yesterday was {_mdy(today - datetime.timedelta(days=1))}."
    elif form == 2:
        lead = f"Tomorrow is {_mdy(today + datetime.timedelta(days=1))}."
    elif form == 3:
        lead = f"Today is {today.strftime('%B')} {today.day}, {today.year}."
    elif form == 4:
        lead = ("In the UK, people usually put the day before the month when formatting the date. Therefore, today is "
                f"{today.strftime('%d/%m/%Y')} to them.")
    else:
        lead = (f"The meeting was planned for {_mdy(today - datetime.timedelta(days=1))}, but it was postponed by one "
                "day to today.")

    def months(d, k):
        m = d.month - 1 + k
        try:
            return d.replace(year=d.year + m // 12, month=m % 12 + 1)
        except ValueError:  # no such day in that month
            return None

    def years(d, k):
        try:
            return d.replace(year=d.year + k)
        except ValueError:  # 29 February
            return None
    shifts = {"tomorrow": lambda d: d + datetime.timedelta(days=1),
              "yesterday": lambda d: d - datetime.timedelta(days=1),
              "one week ago from today": lambda d: d - datetime.timedelta(days=7),
              "one week from today": lambda d: d + datetime.timedelta(days=7),
              "10 days ago": lambda d: d - datetime.timedelta(days=10),
              "24 hours later": lambda d: d + datetime.timedelta(days=1),
              "one year ago from today": lambda d: years(d, -1),
              "a month ago": lambda d: months(d, -1), "today": lambda d: d}
    ask = rng.choice(list(shifts))
    right = shifts[ask](today)
    wrong = {years(right, k) for k in (-1, 1, rng.choice([-40, -20, 30, 50]))}
    wrong |= {right + datetime.timedelta(days=k) for k in (-1, 1, 7, -7)} | {months(right, rng.choice([-1, 1]))}
    if right.day <= 12 and right.day != right.month:
        wrong.add(right.replace(month=right.day, day=right.month))
    wrong -= {right, None}
    choices = [_mdy(d) for d in rng.sample(sorted(wrong), 5)] + [_mdy(right)]
    rng.shuffle(choices)
    return _choice(f"{lead} What is the date {ask} in MM/DD/YYYY?", choices, _mdy(right))


def _clock(h):
    return f"{(h - 1) % 12 + 1}{'am' if h < 12 else 'pm'}"


ERRANDS = ["buying a phone at the electronics store", "walking in the garden", "waiting at the airport",
           "working at the office", "playing tennis at the tennis court", "getting a coffee at the cafe",
           "sitting on a rooftop", "stretching at a yoga studio", "taking photos near the lighthouse",
           "buying bread at the bakery", "reading at the library", "watching a movie at the theater",
           "fixing their bike at the repair shop", "driving to the museum"]
PLACES = ["bookstore", "swimming pool", "art studio", "football field", "orchestra hall", "dentist", "park",
          "gym", "bakery", "post office", "beach", "market"]


def temporal_sequences(rng):
    who, place = rng.choice(NAMES), rng.choice(PLACES)
    wake, close = rng.randint(5, 10), rng.randint(17, 22)
    k = rng.randint(3, 6)
    cuts = sorted(rng.sample(range(wake + 1, close), k))
    spans = list(zip([wake] + cuts, cuts + [close]))
    free = rng.randrange(len(spans))
    witnesses = rng.sample([x for x in NAMES if x != who], len(spans))
    lines = [f"{who} woke up at {_clock(wake)}."]
    lines += [f"{w} saw {who} {e} from {_clock(a)} to {_clock(b)}."
              for i, ((a, b), w, e) in enumerate(zip(spans, witnesses, rng.sample(ERRANDS, len(spans)))) if i != free]
    lines.append(f"The {place} was closed after {_clock(close)}.")
    shown = [free] + rng.sample([i for i in range(len(spans)) if i != free], 3)
    choices = [f"{_clock(spans[i][0])} to {_clock(spans[i][1])}" for i in shown]
    rng.shuffle(choices)
    text = (f"Today, {who} went to the {place}. Between what times could they have gone?\nWe know that:\n"
            + "\n".join(lines) + f"\nBetween what times could {who} have gone to the {place}?")
    return _choice(text, choices, f"{_clock(spans[free][0])} to {_clock(spans[free][1])}")


# --- tables, colored objects ----------------------------------------------------------------------------------------

PENGUIN_NAMES = "Otto Pia Rufus Mabel Ingo Tilda Bruno Greta Felix Hilde Oskar Lotte Emil Frida".split()


def penguins_in_a_table(rng):
    names = rng.sample(PENGUIN_NAMES, 5)
    rows = list(zip(names[:4], rng.sample(range(2, 15), 4), rng.sample(range(40, 100, 5), 4),
                    rng.sample(range(8, 25), 4)))
    a = rows[0]
    text = ("Here is a table where the first line is a header and each subsequent line is a penguin:  name, age, "
            "height (cm), weight (kg) " + " ".join(f"{n}, {x}, {h}, {w}" for n, x, h, w in rows)
            + f"  For example: the age of {a[0]} is {a[1]}, the weight of {rows[1][0]} is {rows[1][3]} kg, the height "
            f"of {rows[2][0]} is {rows[2][2]} cm.  ")
    if rng.random() < 0.3:
        new = (names[4], rng.choice([x for x in range(2, 15) if x not in [r[1] for r in rows]]),
               rng.choice([x for x in range(40, 100, 5) if x not in [r[2] for r in rows]]),
               rng.choice([x for x in range(8, 25) if x not in [r[3] for r in rows]]))
        rows.append(new)
        text += f"We now add a penguin to the table:\n{', '.join(map(str, new))}\n"
    elif rng.random() < 0.3:
        gone = rng.choice(rows)
        rows.remove(gone)
        text += f"We then delete the penguin named {gone[0]} from the table.\n"
    col = rng.choice([(1, "oldest", "youngest"), (2, "tallest", "shortest"), (3, "heaviest", "lightest")])
    form = rng.randrange(4)
    if form == 0:
        top = rng.random() < 0.5
        pick = (max if top else min)(rows, key=lambda r: r[col[0]])
        return _choice(text + f"Which is the {col[1] if top else col[2]} penguin?", names, pick[0])
    if form == 1:
        order = sorted(r[0] for r in rows)
        last = rng.random() < 0.5
        return _choice(text + f"What is the name of the {'last' if last else 'first'} penguin sorted by alphabetic "
                       "order?", names, order[-1] if last else order[0])
    if form == 2:
        t = rng.choice(sorted({r[1] for r in rows}))
        more = rng.random() < 0.5
        k = sum((r[1] > t) if more else (r[1] < t) for r in rows)
        return _choice(text + f"How many penguins are {'more' if more else 'less'} than {t} years old?",
                       [str(i) for i in range(1, 6)] if k else [str(i) for i in range(5)], str(k))
    total = sum(r[1] for r in rows)
    choices = sorted({total} | set(rng.sample([total + d for d in (-7, -5, -3, -2, -1, 1, 2, 3, 5, 7) if total + d > 0],
                                               4)))
    return _choice(text + "What is the cumulated age of the penguins?", [str(c) for c in choices], str(total))


COLORS = ("red orange yellow green blue brown magenta fuchsia mauve teal turquoise burgundy silver gold black grey "
          "purple pink").split()
THINGS = {"cup": "cups", "mug": "mugs", "teddy bear": "teddy bears", "pen": "pens", "pencil": "pencils",
          "notebook": "notebooks", "keychain": "keychains", "sheet of paper": "sheets of paper", "booklet": "booklets",
          "envelope": "envelopes", "fidget spinner": "fidget spinners", "stress ball": "stress balls",
          "paperclip": "paperclips", "pair of sunglasses": "pairs of sunglasses", "phone charger": "phone chargers",
          "crayon": "crayons", "puzzle": "puzzles", "dog leash": "dog leashes", "cat toy": "cat toys",
          "plate": "plates", "bracelet": "bracelets", "necklace": "necklaces"}
SURFACES = ["On the desk", "On the floor", "On the table", "On the nightstand"]


def reasoning_about_colored_objects(rng):
    surface = rng.choice(SURFACES)
    form = rng.randrange(3)
    if form < 2:  # a row of distinct things
        things = rng.sample(list(THINGS), rng.randint(3, 6))
        colors = [rng.choice(COLORS) for _ in things]
        row = _list([f"{_a(c)} {t}" for c, t in zip(colors, things)])
        if form == 0:
            i = rng.randrange(len(things) - 1)
            right = rng.random() < 0.5
            ref, ans = (i, i + 1) if right else (i + 1, i)
            q = (f"What is the color of the thing directly to the {'right' if right else 'left'} of the "
                 f"{things[ref]}?")
            return _choice(f"{surface}, you see a set of things arranged in a row: {row}. {q}", COLORS, colors[ans])
        i = rng.randrange(len(things))
        k = sum(c != colors[i] for c in colors[i + 1:])
        q = f"How many non-{colors[i]} things do you see to the right of the {things[i]}?"
        return _choice(f"{surface}, you see a set of things arranged in a row: {row}. {q}", NUMBER_WORDS[:7],
                       NUMBER_WORDS[k])
    kinds, palette = rng.sample(list(THINGS), 2), rng.sample(COLORS, 4)
    piles = {}
    for _ in range(rng.randint(4, 7)):
        piles[(rng.choice(palette), rng.choice(kinds))] = rng.randint(1, 3)
    listing = _list([f"{NUMBER_WORDS[n]} {c} {kind if n == 1 else THINGS[kind]}" for (c, kind), n in piles.items()])
    gone = rng.choice(kinds)
    color = rng.choice(sorted({c for c, _ in piles}))
    left = sum(n for (c, kind), n in piles.items() if kind != gone and c == color)
    q = f"If I remove all the {THINGS[gone]} from the {surface.split()[-1]}, how many {color} things remain on it?"
    return _choice(f"{surface.split()[0]} the {surface.split()[-1]}, there is {listing}. {q}", NUMBER_WORDS[:17],
                   NUMBER_WORDS[left])


# --- adjectives, shapes ---------------------------------------------------------------------------------------------

ADJECTIVES = [  # the conventional English order: opinion, size, age, shape, color, origin, material, purpose
    "lovely ugly obnoxious wonderful terrible silly nice awful good repulsive mysterious fine".split(),
    "big small tiny huge enormous little large massive midsize".split(),
    "old new ancient young brand-new old-fashioned archaic".split(),
    "square circular triangular rectangular spherical prismlike pyramidal".split(),
    "red blue green yellow black white grey brown purple pink gray".split(),
    "American Chinese Mexican Brazilian Indian German Japanese Turkish Egyptian Nigerian Thai Russian".split(),
    "wool plastic glass leather paper rubber steel cloth cardboard wood fiberglass lead iron gold silver".split(),
    "smoking hiking walking drinking eating typing exercise snorkeling whittling".split(),
]
NOUNS = "sweater knife ship shoe chair box bag car table cat dog spoon watch bowl monkey surfboard".split()


def hyperbaton(rng):
    cats = sorted(rng.sample(range(len(ADJECTIVES)), rng.randint(2, 5)))
    words = [rng.choice(ADJECTIVES[c]) for c in cats]
    noun = rng.choice(NOUNS)
    while True:
        bad = rng.sample(words, len(words))
        if bad != words:
            break
    right, wrong = " ".join(words + [noun]), " ".join(bad + [noun])
    choices = [right, wrong] if rng.random() < 0.5 else [wrong, right]
    return _choice("Which sentence has the correct adjective order:", choices, right)


SHAPES = ["circle", "heptagon", "hexagon", "kite", "line", "octagon", "pentagon", "rectangle", "sector", "triangle"]


def geometric_shapes(rng):
    import math  # noqa: PLC0415
    shape = rng.choice(SHAPES[1:] + ["trapezoid", "ellipse", "circle"])
    cx, cy = rng.uniform(25, 75), rng.uniform(25, 75)

    def pt(p):
        return f"{p[0]:.2f},{p[1]:.2f}"

    def path(points, broken):
        if broken and len(points) > 2:  # each side drawn on its own
            segs = list(zip(points, points[1:] + points[:1]))
            return " ".join(f"M {pt(a)} L {pt(b)}" for a, b in segs)
        return "M " + " L ".join(pt(p) for p in points + points[:1])
    if shape in ("triangle", "pentagon", "hexagon", "heptagon", "octagon"):
        k = {"triangle": 3, "pentagon": 5, "hexagon": 6, "heptagon": 7, "octagon": 8}[shape]
        r, t0 = rng.uniform(12, 30), rng.uniform(0, 2 * math.pi)
        angles = sorted(t0 + 2 * math.pi * (i + rng.uniform(-0.2, 0.2)) / k for i in range(k))
        d = path([(cx + r * math.cos(a), cy + r * math.sin(a)) for a in angles], rng.random() < 0.4)
    elif shape == "line":
        a = rng.uniform(0, 2 * math.pi)
        r = rng.uniform(10, 30)
        d = f"M {pt((cx, cy))} L {pt((cx + r * math.cos(a), cy + r * math.sin(a)))}"
    elif shape in ("rectangle", "kite", "trapezoid"):
        a, w, h = rng.uniform(0, math.pi), rng.uniform(10, 30), rng.uniform(10, 30)
        if shape == "rectangle":
            while abs(w - h) < 3:
                h = rng.uniform(10, 30)
            local = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
        elif shape == "kite":
            top, bottom = rng.uniform(5, 12), rng.uniform(15, 30)
            local = [(0, -top), (w / 2, 0), (0, bottom), (-w / 2, 0)]
        else:
            short = rng.uniform(0.3, 0.7) * w
            shift = rng.uniform(-0.2, 0.2) * w
            local = [(-w / 2, -h / 2), (w / 2, -h / 2), (shift + short / 2, h / 2), (shift - short / 2, h / 2)]
        pts = [(cx + x * math.cos(a) - y * math.sin(a), cy + x * math.sin(a) + y * math.cos(a)) for x, y in local]
        d = path(pts, rng.random() < 0.3)
    elif shape == "sector":
        r, a = rng.uniform(15, 35), rng.uniform(0, 2 * math.pi)
        b = a + rng.uniform(0.4, 2.5)
        p, q = (cx + r * math.cos(a), cy + r * math.sin(a)), (cx + r * math.cos(b), cy + r * math.sin(b))
        d = f"M {pt((cx, cy))} L {pt(p)} A {r:.2f},{r:.2f} 0.00 0,1 {pt(q)} L {pt((cx, cy))}"
    else:  # circle, ellipse: two half arcs between opposite points
        rx = rng.uniform(10, 30)
        ry = rx if shape == "circle" else rx * rng.choice([rng.uniform(0.4, 0.75), rng.uniform(1.35, 2.0)])
        rot = rng.uniform(0, 360)
        t = math.radians(rot)
        p, q = (cx + rx * math.cos(t), cy + rx * math.sin(t)), (cx - rx * math.cos(t), cy - rx * math.sin(t))
        d = f"M {pt(p)} A {rx:.2f},{ry:.2f} {rot:.2f} 1,0 {pt(q)} A {rx:.2f},{ry:.2f} {rot:.2f} 1,0 {pt(p)}"
    extra = [] if shape not in ("trapezoid", "ellipse") and rng.random() < 0.8 else \
        [shape if shape in ("trapezoid", "ellipse") else rng.choice(["trapezoid", "ellipse"])]
    if shape == "rectangle" and extra == ["trapezoid"]:  # a rectangle is a trapezoid too
        extra = ["ellipse"]
    if shape == "circle" and extra == ["ellipse"]:  # and a circle an ellipse
        extra = ["trapezoid"]
    return _choice(f'This SVG path element <path d="{d}"/> draws a', SHAPES + extra, shape)


# --- arguments ------------------------------------------------------------------------------------------------------

RELATIONS = ["a cousin of {}", "a classmate of {}", "an ancestor of {}", "a stepbrother of {}", "a half-sister of {}",
             "a workmate of {}", "a grandson of {}", "a close friend of {}", "an uncle of {}", "a niece of {}"]
CLUBS = ["a fan of {}", "a supporter of {}", "a critic of {}", "an expert of {}", "a former member of {}",
         "a devotee of {}"]
CLUB_NAMES = ["Harbor City FC", "Redwood Rovers", "Atletico Lumen", "Nordvik United", "Sporting Calderon",
              "Valemont Athletic", "FC Brightwater", "Kestrel Town", "Union Ostara", "Dynamo Velk"]


def _predicates(rng):
    if rng.random() < 0.5:
        return [r.format(rng.choice(NAMES)) for r in rng.sample(RELATIONS, 4)]
    return [r.format(c) for r, c in zip(rng.sample(CLUBS, 4), rng.sample(CLUB_NAMES, 4))]


def formal_fallacies(rng):
    """A short argument of universal premises and facts about one person; valid iff no assignment of the predicates
    to a single individual makes every premise true and the conclusion false (monadic: one individual suffices)."""
    preds = _predicates(rng)
    who = rng.choice([x for x in NAMES if not any(x in p for p in preds)])
    want = rng.random() < 0.5

    def lit(i, neg):
        return ("not " if neg else "") + preds[i]

    def being(i, neg):
        return ("not being " if neg else "being ") + preds[i]

    def rule(rng_):
        k = rng_.choice([1, 1, 2])
        ante = [(i, rng_.random() < 0.3) for i in rng_.sample(range(4), k)]
        cons = (rng_.choice([i for i in range(4) if i not in [a for a, _ in ante]]), rng_.random() < 0.35)
        return ante, cons

    def rule_text(r):
        ante, (c, cn) = r
        if len(ante) == 1:
            (a, an), = ante
            s = rng.choice([f"Whoever is {lit(a, an)} is {lit(c, cn)}.",
                            f"{being(a, an)} is sufficient for {being(c, cn)}.",
                            f"{being(c, cn)} is necessary for {being(a, an)}.",
                            f"Everyone who is {lit(a, an)} is {lit(c, cn)}, too."])
            return s[0].upper() + s[1:]
        (a, an), (b, bn) = sorted(ante, key=lambda x: x[1])  # a negated literal never first: "a X and not a Y"
        both = f"neither {preds[a]} nor {preds[b]}" if an and bn else f"{lit(a, an)} and {lit(b, bn)}"
        return f"Whoever is {both} is also {lit(c, cn)}."

    def rule_holds(r, v):
        ante, (c, cn) = r
        return not all(v[a] != an for a, an in ante) or v[c] != cn
    while True:
        rules = [rule(rng) for _ in range(rng.randint(1, 2))]
        about_person = rng.random() < 0.5
        facts = [(i, rng.random() < 0.3) for i in rng.sample(range(4), rng.randint(1, 2))] if about_person else []
        if about_person:
            concl = (rng.randrange(4), rng.random() < 0.4)
            if concl[0] in [i for i, _ in facts]:
                continue
        else:
            concl = rule(rng)
            if concl in rules:  # a premise restated
                continue
        valid = True
        for v in itertools.product([False, True], repeat=4):
            prem = all(rule_holds(r, v) for r in rules) and all(v[i] != n for i, n in facts)
            if about_person:
                ok = v[concl[0]] != concl[1]
            else:
                ok = rule_holds(concl, v)
            if prem and not ok:
                valid = False
                break
        if valid == want:
            break
    premises = [rule_text(r) for r in rules] + [f"{who} is {lit(i, n)}." for i, n in facts]
    rng.shuffle(premises)
    lead = ["First", "To begin with", "To start with"]
    more = ["Next", "Moreover", "Second", "Plus", "Besides"]
    body = f"{rng.choice(lead)}, {premises[0][0].lower()}{premises[0][1:]}"
    for p in premises[1:]:
        body += f" {rng.choice(more)}, {p[0].lower()}{p[1:]}" if not p.startswith(who) else f" {rng.choice(more)}, {p}"
    conclusion = f"{who} is {lit(*concl)}." if about_person else rule_text(concl)
    if not conclusion.startswith(who):
        conclusion = conclusion[0].lower() + conclusion[1:]
    frame = rng.choice(["Here comes an argument", "Consider the following argument", "Someone puts forward this "
                        "argument", "Here is an argument about people and their relations"])
    text = (f'"{frame}: {body} {rng.choice(["Therefore", "Hence", "It follows that", "So"])}, {conclusion}"\nIs the '
            "argument, given the explicitly stated premises, deductively valid or invalid?\nOptions:\n- valid \n"
            "- invalid")
    return text, "valid" if valid else "invalid"


# --- the set --------------------------------------------------------------------------------------------------------

def generators():
    """BBH task name -> rng -> (problem, answer)."""
    vocab = _vocabulary()
    gens = {"boolean_expressions": boolean_expressions, "date_understanding": date_understanding,
            "formal_fallacies": formal_fallacies, "geometric_shapes": geometric_shapes, "hyperbaton": hyperbaton,
            "multistep_arithmetic_two": multistep_arithmetic, "navigate": navigate, "object_counting": object_counting,
            "penguins_in_a_table": penguins_in_a_table,
            "reasoning_about_colored_objects": reasoning_about_colored_objects,
            "temporal_sequences": temporal_sequences, "web_of_lies": web_of_lies,
            "word_sorting": lambda rng: word_sorting(rng, vocab)}
    for n, w in ((3, "three"), (5, "five"), (7, "seven")):
        gens[f"logical_deduction_{w}_objects"] = lambda rng, n=n: logical_deduction(rng, n)
        gens[f"tracking_shuffled_objects_{w}_objects"] = lambda rng, n=n: tracking_shuffled_objects(rng, n)
    return dict(sorted(gens.items()))


def make_bbhsynth(per_task=128, seed=0):
    """Write data/practice/bbhsynth.jsonl.gz: per_task distinct items of every generated kind, none a benchmark
    question."""
    from .data import PRACTICE_DIR, bench_fingerprints, fingerprint  # noqa: PLC0415
    fps, rows = bench_fingerprints(), []
    for task, gen in generators().items():
        rng, seen = random.Random(f"{seed}:{task}"), set()
        while len(seen) < per_task:
            problem, answer = gen(rng)
            f = fingerprint(problem)
            if f in fps or f in seen:
                continue
            seen.add(f)
            rows.append({"id": f"bbhsynth_{task}_{len(seen) - 1}", "source": "bbhsynth", "task": "exact",
                         "subject": task, "problem": problem, "answer": answer})
    write_jsonl(PRACTICE_DIR / "bbhsynth.jsonl.gz", rows)
    return rows
