from vector_stuff.vectorizer import Vectorizer
from vector_stuff.tokenizer import Tokenizer
from matplotlib import pyplot as plt
import os, json


RATING_PER_REQ = 3
NAMES = ["rater 1", "axel", "rater 3", "jihad", "nadav", "rater 2"]
NAMES = ["rater 1", "rater 2", "rater 3"]
PROMPTS = ["zeroshot", "fewshot"]  # , "roleplay", "role_and_few"]
RATINGS = ["False negative", "True positive"]
STANDARDS = ["ISO_8528_8_2016_EN", "SS_EN_60871_1_EN", "IEC_IEEE_60214"]
RUNS = ["llm", "bing"]
HIGHLIGHT_WINDOW = 5

TESTING = False

folder = "trial_240415"

"""
!! MUST BE RUN FROM /INPUT_OUTPUT !!
"""

vectorizer = Vectorizer()
tokenizer = Tokenizer()


def color_text(text, match_value):
    color_intensity = int((match_value / 100) * 255)
    color_code = f"\033[38;2;{255-color_intensity};{255};{255-color_intensity}m"
    reset_code = "\033[0m"
    return color_code + text + reset_code


def similarity_printer(llm_output, manual_output):
    # handle manual for comparison
    manual_tokens = tokenizer.tokenize_sentences(manual_output)
    # manual_vector = vectorizer.vectorize_tokens(manual_tokens)
    manual_vectors = []
    token_length = len(manual_tokens)
    for i in range(token_length - HIGHLIGHT_WINDOW):
        manual_vector = vectorizer.vectorize_tokens(
            manual_tokens[i : i + HIGHLIGHT_WINDOW]
        )
        manual_vectors.append(manual_vector)

    # handle all sentences from llm output
    llm_objects = []
    llm_sentences = tokenizer.extract_sentences(llm_output)
    for index, text_senence in enumerate(llm_sentences):
        token_sentence = tokenizer.tokenize_sentences(text_senence)
        vector_sentence = vectorizer.vectorize_tokens(token_sentence)

        dist = 10**10
        for mv in manual_vectors:
            d = vectorizer.euclidean_distance(mv, vector_sentence)
            if d < dist:
                dist = d

        sent_object = {
            "index": index,
            "text": text_senence,
            "tokens": token_sentence,
            "vector": vector_sentence,
            "vector_dist": dist,
            "green_intensity": 0,
        }
        llm_objects.append(sent_object)

    # order by similarity
    if TESTING:
        llm_objects.sort(key=lambda x: x["vector_dist"])
        plt.plot([obj["vector_dist"] for obj in llm_objects])

    # closest sentence distance
    min_dist = llm_objects[0]["vector_dist"]
    max_color_dist = 1.1 * min_dist
    min_dist = 1.5
    max_color_dist = 2.5

    dx = max_color_dist - min_dist
    dy = -1
    k = dy / dx
    m = 1 - k * min_dist

    for obj in llm_objects:
        if obj["vector_dist"] < max_color_dist:
            obj["green_intensity"] = int(100 * (k * obj["vector_dist"] + m))
        if obj["vector_dist"] < min_dist:
            obj["green_intensity"] = 100

    # order by similarity
    llm_objects.sort(key=lambda x: x["vector_dist"])
    for obj in llm_objects:
        print(
            f"Index: {obj['index']}, Distance: {obj['vector_dist']}, Intensity: {obj['green_intensity']}"
        )

    print("--------- LLM OUTPUT ----------")
    # order by index for printing
    colored_output = ""
    llm_objects.sort(key=lambda x: x["index"])
    for obj in llm_objects:
        # print with intensity of green

        original = obj["text"]
        colored = color_text(obj["text"], obj["green_intensity"])

        # find the matching sentence
        llm_output = llm_output.replace(original, colored)

    print(llm_output)
    print("")

    print("----- MANUALLY EXTRACTED  -----")
    print(color_text(manual_output, 100))


total_reqs = 0
for file_name in STANDARDS:
    doc_json = json.load(
        open(folder + "/backup_GS/" + file_name + ".json", "r", encoding="utf-8")
    )
    for page in doc_json["script"]:
        total_reqs += 1
print(f"Total requirements: {total_reqs}")

stride = total_reqs / len(NAMES)
print(f"Stride: {stride}")

reqs_per_rater = stride * RATING_PER_REQ

order = []
index = 0
for file_name in STANDARDS:
    doc_json = json.load(
        open(folder + "/backup_GS/" + file_name + ".json", "r", encoding="utf-8")
    )
    for req in doc_json["script"]:
        order.append((index, {"doc": file_name, "id": req["id"], "page": req["page"]}))
        index += 1
for entry in order:
    print(entry)

distribution = []

strides_taken = 0
strides_rounded = 0
for rater in NAMES:
    for i in range(int(reqs_per_rater)):
        index, req = order[(strides_rounded + i) % len(order)]
        json_object = {
            "rater": rater,
            "doc": req["doc"],
            "id": req["id"],
            "page": req["page"],
        }
        distribution.append(json_object)
    strides_taken += stride
    strides_rounded = round(strides_taken)

for name in NAMES:
    print(f"Rater: {name}")
    for entry in distribution:
        if entry["rater"] == name:
            print(entry)

# count duplicates on doc and id, should be 3
duplicates = []
for entry in distribution:
    doc = entry["doc"]
    id = entry["id"]
    json_object = {"doc": doc, "id": id}
    found = False
    for index, (count, doc_and_id) in enumerate(duplicates):
        if doc_and_id == json_object:
            duplicates[index] = (count + 1, doc_and_id)
            found = True
            break
    if not found:
        duplicates.append((1, json_object))

for count, doc_and_id in duplicates:
    if count != RATING_PER_REQ:
        print(f"Doc: {doc_and_id['doc']}, id: {doc_and_id['id']}, count: {count}")

# choose rater
for index, rater in enumerate(NAMES):
    print(f"{index}: {rater}")
print("'q' to quit")
while True:
    try:
        value = input("Choose rater: ")
        rater = NAMES[int(value)]
        break
    except:
        if value == "q":
            exit()
        print("Invalid input, enter index again")
print(f"Rater: {rater}")

# choose prompting type
for index, prompt in enumerate(PROMPTS):
    print(f"{index}: {prompt}")
print("'q' to quit")
while True:
    try:
        value = input("Choose prompt: ")
        prompt_type = PROMPTS[int(value)]
        break
    except:
        if value == "q":
            exit()
        print("Invalid input, enter index again")
print(f"Prompt: {prompt_type}")

# choose llm or bing
for index, run in enumerate(RUNS):
    print(f"{index}: {run}")
print("'q' to quit")
while True:
    try:
        value = input("Choose run: ")
        run = RUNS[int(value)]
        break
    except:
        if value == "q":
            exit()
        print("Invalid input, enter index again")

# get raters work
folder_path = run + "_rated/"
rater_work = [entry for entry in distribution if entry["rater"] == rater]
print(f"Rater work: {len(rater_work)}")
for entry in rater_work:
    print(entry)

# rate the requirements true possitive or false negative

for entry in rater_work:
    print("------------REQUIREMENT------------")
    print(f"Doc: {entry['doc']}, page: {entry['page']}, id: {entry['id']}")
    print("")

    doc = entry["doc"]
    req_index = entry["id"]
    page = entry["page"]
    file_name = doc.split("/")[-1].split(".")[0]
    doc_path = folder + "/" + folder_path + file_name + ".json"
    doc_json = json.load(open(doc_path, "r", encoding="utf-8"))
    current_req = [data for data in doc_json["script"] if data["id"] == req_index]

    # print("------------LLM OUTPUT------------")
    llm_doc_path = folder + "/" + run + "_extractions/" + file_name + ".json"
    llm_doc_json = json.load(open(llm_doc_path, "r", encoding="utf-8"))
    extraction_for_prompt_type = llm_doc_json["extractions"]
    for extraction in extraction_for_prompt_type:
        if extraction["page"] == page:
            for data in extraction["data"]:
                if data["prompt"] == prompt_type + ".txt":
                    # print(data["output"])
                    # print("")
                    llm_print = data["output"]

    # print("------------REQUIREMENT------------")
    # print(current_req[0]["text"])
    # print("")
    req_print = current_req[0]["text"]

    # fancy ass similarity printer
    similarity_printer(llm_print, req_print)

    for index, rating in enumerate(RATINGS):
        print(f"{index}: {rating}")
    print("'q' to quit")
    if not TESTING:
        while True:
            try:
                value = input("Choose rating: ")
                rating = RATINGS[int(value)]
                break
            except:
                if value == "q":
                    exit()
                print("Invalid input, enter index again")

        # write back to doc_json with req_id

        for obj in doc_json["script"]:
            if obj["id"] == req_index:
                if rating == "True positive":
                    obj["ratings"][prompt_type][rater] = "TP"
                elif rating == "False negative":
                    obj["ratings"][prompt_type][rater] = "FN"
                break

        with open(doc_path, "w", encoding="utf-8") as f:
            json.dump(doc_json, f, indent=2)


# name the axis
if TESTING:
    plt.xlabel("req")
    plt.ylabel("distance")
    plt.show()
    exit()
# rate the requirements false positive
distinct_docs_and_pages = []
for entry in rater_work:
    doc = entry["doc"]
    page = entry["page"]
    if (doc, page) not in distinct_docs_and_pages:
        distinct_docs_and_pages.append((doc, page))

for doc, page in distinct_docs_and_pages:
    print("------------DOCUMENT------------")
    print(f"Doc: {doc}, page: {page}")
    print("")

    llm_doc_path = r"C:\Users\willi\RequirementsAndPromptEngineeringThesis\first_result\bing_answers\zero_shot_SS_EN_60871.json"
    llm_doc_json = json.load(open(llm_doc_path, "r", encoding="utf-8"))
    print(llm_doc_json)
    # select output for current prompt type
    extraction_for_prompt_type = llm_doc_json["extractions"]
    for extraction in extraction_for_prompt_type:
        if extraction["page"] == page:
            for data in extraction["data"]:
                if data["prompt"] == prompt_type + ".txt":
                    print("------------LLM OUTPUT------------")
                    print(data["output"])
                    print("")

    # print all manual extractions
    doc_path = folder + "/" + folder_path + doc.split("/")[-1].split(".")[0] + ".json"
    doc_json = json.load(open(doc_path, "r", encoding="utf-8"))
    current_req = [data for data in doc_json["script"] if data["page"] == page]
    for index, req in enumerate(current_req):
        print(f"------------REQUIREMENT {index}------------")
        print(req["text"])
        print("")

    print("COUNT FALSE POSITIVES")
    while True:
        try:
            value = input("Manual count, q for exit: ")
            if int(value) >= 0:
                count = int(value)
                break
        except:
            if value == "q":
                exit()
            print("Invalid input")
    # TODO add page key
    doc_json["FP"][prompt_type][str(page)].append({"rater": rater, "count": count})

    with open(doc_path, "w", encoding="utf-8") as f:
        json.dump(doc_json, f, indent=2)
