import argparse, os, json, datetime
from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple
from langchain.callbacks import FileCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from loguru import logger
from lib import *
import numpy as np
import pandas as pd

current_folder = "input_output/trial_240429"


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def get_pages(available_pages: int):
    print("Available pages: ", available_pages)
    pages = input("Enter the pages you want to extract from (comma separated)\n>>> ")
    return [int(i) for i in pages.split(",")]


def looper():
    # input file
    docs_and_pages_path = "input_output/docs_and_pages.json"
    docs_and_pages = json.load(open(docs_and_pages_path, "r"))

    # fdsfsdfsdfsdfsdfdsfdsfdsfsdfdsfdsfsdfsdfsdfds
    # output folder with time to avoid overwriting
    output_folder_path = current_folder + "/llm_extractions"
    # os.makedirs(output_folder_path)

    # get templates
    templates: list[(str, PromptTemplate)] = get_templates_plural()
    # only keep fewshot.txt
    # templates = [t for t in templates if "fewshot" in t[0]]

    # init LLM
    llm: LLM = LLM()

    # loop over documents
    for dnp in docs_and_pages:
        # load the pdf
        loaded: PDFLoader = PDFLoader(dnp["path"])

        # init output json with document path and pages
        output: json = {}
        output["document"] = dnp["path"]
        output["pages"] = dnp["pages"]

        # list of json objects per page
        extractions_per_page: list[dict] = []
        # loop over pages
        for page in dnp["pages"]:
            print("Working on page: ", page)
            # list of json objects per template
            extractions_per_template: list[dict] = []
            # loop over templates
            for prompt_type, template in templates:

                # init output json with template
                chain = RequirementExtractor(llm=llm, prompt_template=template)

                # load single page
                doc: Document = list(
                    filter(lambda x: x.metadata["page"] == page - 1, loaded.documents)
                )[
                    0
                ]  # zero indexed

                # invoke chain with template and page
                llm_output = chain.invoke(input=doc)
                llm_output_string = llm_output["text"]
                llm_output_json = {"prompt": prompt_type, "output": llm_output_string}
                extractions_per_template.append(llm_output_json)

            # add data per page to output
            json_per_page = {"page": page, "data": extractions_per_template}
            extractions_per_page.append(json_per_page)

        # add data per page to output
        output["extractions"] = extractions_per_page

        # output to file
        output_path = (
            output_folder_path
            + "/"
            + dnp["path"].split("/")[-1].split(".")[0]
            + ".json"
        )
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)


def print_pages(doc, page):
    print("Prining page: ", page, " from: ", doc)
    text = PDFLoader(doc).documents[page - 1]
    # pritty print
    print(text.page_content)
    pass


def metrics(result_folder, llm):
    prompt_types = ["zeroshot", "fewshot", "roleplay", "role_and_few"]
    prompt_types = ["zeroshot", "fewshot"]
    rated_path = result_folder + f"/{llm}_rated"
    files = os.listdir(rated_path)
    for file in files:
        print("Working on: ", file)
        rated = json.load(open(rated_path + "/" + file, "r"))
        for prompt in prompt_types:
            print("    Prompt type: ", prompt)
            tp = 0
            fn = 0
            fp = 0
            # count TP and FN for each req
            for entry in rated["script"]:
                ratings = entry["ratings"][prompt]
                """
                {'arthur': 'FN', 'william': 'FN', 'axel': 'FN'}
                """
                # count all TP
                tp += list(ratings.values()).count("TP")
                # count all FN
                fn += list(ratings.values()).count("FN")

            tp /= 3
            fn /= 3

            rated_fp = rated["FP"][prompt]
            for entry in rated_fp.keys():
                length = len(rated_fp[entry])
                # sum of count
                local_fp = 0
                for obj in rated_fp[entry]:
                    local_fp += obj["count"]
                local_fp = local_fp / length
                fp += local_fp

            print(
                "\tTP:\t\t{:.2f} \n\tFN:\t\t{:.2f} \n\tFP:\t\t{:.2f}".format(tp, fn, fp)
            )
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            f1 = 2 * (precision * recall) / (precision + recall)
            print(
                "\tRecall:\t\t{:.2f} \n\tPrecision:\t{:.2f} \n\tF1:\t\t{:.2f}".format(
                    recall, precision, f1
                )
            )


def metrics2(result_folder, llm_type):
    docs_and_pages_path = "input_output/docs_and_pages.json"
    with open(docs_and_pages_path, "r") as f:
        docs_and_pages = json.loads(f.read())

    res = {}

    def filter_unagreement(prompt_type_result: dict):
        assert len(prompt_type_result) == 3, "This is a bad entry! (not 3 raters)"

        tp = fn = 0
        for rater, rating in prompt_type_result.items():
            match rating:
                case "TP":
                    tp += 1
                case "FN":
                    fn += 1
                case _:
                    raise AssertionError(
                        "Wrong value in prompt type result: "
                        + f"{json.dumps(prompt_type_result, indent=2)}"
                    )

        delta = np.abs(tp - fn)
        if delta == 3 or delta == 2:
            return "TP" if tp > fn else "FN"
        return np.nan

    prompt_types = ["zeroshot", "fewshot"]

    iec = {}
    iso = {}
    ieee = {}
    for prompt_type in prompt_types:

        iec[prompt_type] = []
        iso[prompt_type] = []
        ieee[prompt_type] = []

    for obj in docs_and_pages:
        name = obj["path"].split("/")[-1][:-4] + ".json"
        rated_path = result_folder + f"/{llm_type}_rated/{name}"
        page_nr = obj["pages"]
        with open(rated_path, "r") as result:
            res = json.loads(result.read())
        FN_TP = list(filter(lambda o: o["page"] in obj["pages"], res["script"]))
        for prompt_type in prompt_types:
            # TP and FN
            tp = 0
            fn = 0
            print("    " + prompt_type)
            FN_TP_prompt_type = list(map(lambda o: o["ratings"][prompt_type], FN_TP))

            for entry in FN_TP_prompt_type:
                # print(entry)

                filtered = filter_unagreement(entry)
                if filtered == "TP":
                    tp += 1
                elif filtered == "FN":
                    fn += 1

            # FP
            fp = 0
            FP = res["FP"][prompt_type]
            fp_list = []
            for entry in FP[f"{page_nr[0]}"]:
                fp_list.append(entry["count"])

            fp = np.median(fp_list)

            print(f"\tTP: {tp}\t\tFN: {fn}\t\tFP: {fp}")
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)

            if precision + recall == 0:
                f1 = 0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)
            print(
                "\tRec.: {:.2f}\tPrec.: {:.2f}\tF1: {:.2f}".format(
                    recall, precision, f1
                )
            )

            temp_data = [
                obj["pages"][0],
                (tp + fn),
                tp,
                fn,
                int(fp),
                f"{recall:.2f}",
                f"{precision:.2f}",
                f"{f1:.2f}",
            ]

            if name == "SS_EN_60871_1_EN.json":
                iec[prompt_type].append(temp_data)
            if name == "ISO_8528_8_2016_EN.json":
                iso[prompt_type].append(temp_data)
            if name == "IEC_IEEE_60214.json":
                ieee[prompt_type].append(temp_data)

            # df = pd.DataFrame(FN_TP_prompt_type).T
            #
            # print(
            #     df.to_latex(index=True)
            #     .replace("\\toprule", "\\hline")
            #     .replace("\\midrule", "\\hline")
            #     .replace("\\bottomrule", "\\hline")
            # )
    print_latex_table(iec, iso, ieee, prompt_types, result_folder, llm_type)


def print_latex_table(iec, iso, ieee, prompt_types, result_folder, llm_type):
    for prompt_type in prompt_types:
        print(f"PROMPT TYPE: {prompt_type}")
        print("IEC Table \n")
        print(result_folder)
        if llm_type == "llm":
            llm_model = r"llama2 7b-q4\_0"
        if llm_type == "bing":
            llm_model = "Copilot"
        table = generate_latex_table(
            data=iec[prompt_type],
            caption=r"IEC 60871-1 \cite{iec60871}",
            label="X",
            prompt_type=prompt_type,
            llm_type=llm_model,
        )
        print(table)
        print("ISO Table \n")
        table = generate_latex_table(
            data=iso[prompt_type],
            caption=r"ISO 8528-8 \cite{iso8528}",
            label="X",
            prompt_type=prompt_type,
            llm_type=llm_model,
        )
        print(table)
        print("IEEE Table \n")
        table = generate_latex_table(
            data=ieee[prompt_type],
            caption=r"IEEE 60214-4 \cite{ieee60214}",
            label="X",
            prompt_type=prompt_type,
            llm_type=llm_model,
        )
        print(table)


def generate_latex_table(data, caption, label, prompt_type, llm_type):
    latex_table = "\\begin{table}[H]\n"
    latex_table += "    \\centering\n"
    latex_table += "    \\begin{tabular}{|c|c|c|c|c|c|c|c|c|}\n"
    latex_table += "        \\hline\n"
    latex_table += "        \\multicolumn{8}{|c|}{" + caption + "}\\\\\n"
    latex_table += "        \\hline\n"
    latex_table += "        \\cellcolor[gray]{0.8}\\textbf{Page} & \\cellcolor[gray]{0.8}\\textbf{GS} & \\cellcolor[gray]{0.8}\\textbf{TP} & \\cellcolor[gray]{0.8}\\textbf{FN} & \\cellcolor[gray]{0.8}\\textbf{FP} & \\cellcolor[gray]{0.8}\\textbf{Recall} & \\cellcolor[gray]{0.8}\\textbf{Precision} & \\cellcolor[gray]{0.8}\\textbf{F1} \\\\\n"
    latex_table += "        \\hline\n"

    for row in data:
        latex_table += "        " + " & ".join([str(x) for x in row]) + " \\\\\n"
        latex_table += "        \\hline\n"

    # Calculate averages
    averages = []
    if data:  # Ensure data is not empty to avoid division by zero
        recalls = [float(row[5]) for row in data]
        precisions = [float(row[6]) for row in data]
        f1s = [float(row[7]) for row in data]
        averages.append(round(sum(recalls) / len(recalls), 2))
        averages.append(round(sum(precisions) / len(precisions), 2))
        averages.append(round(sum(f1s) / len(f1s), 2))

    # Replace placeholders in "Average" row with calculated averages
    latex_table += "        \\textbf{Average} & - & - & - & - & "
    latex_table += " & ".join(map(str, averages)) + " \\\\\n"

    latex_table += "        \\hline\n"
    latex_table += "    \\end{tabular}\n"
    latex_table += (
        "    \\caption{Results from "
        + llm_type
        + " using "
        + prompt_type
        + " on document "
        + caption
        + " }\n"
    )
    latex_table += "    \\label{" + label + "}\n"
    latex_table += "\\end{table}\n"

    return latex_table


def main():
    parser = argparse.ArgumentParser(description="Requirement Extractor")
    parser.add_argument(
        "--path", required=False, type=str, help="Path to the pdf standard"
    )

    parser.add_argument("--loop", "-l", action="store_true", help="Yuuuuuge loop")

    parser.add_argument("--print_pages", "-pp", action="store_true", help="Print pages")
    parser.add_argument("--page", "-p", type=int, help="Page to extract from")
    parser.add_argument("--doc", "-d", type=str, help="Document to extract from")

    parser.add_argument("--metrics", "-m", action="store_true", help="Confusion matrix")
    parser.add_argument(
        "--metrics2", "-m2", action="store_true", help="New metrics calc"
    )
    parser.add_argument("--result_folder", "-rf", type=str, help="Folder with jsons")
    parser.add_argument("-llm", type=str, help="bing or llm")

    args = parser.parse_args()

    if args.loop:
        looper()
        exit()

    if args.print_pages:
        print_pages(args.doc, args.page)
        exit()

    if args.metrics:
        metrics(args.result_folder, args.llm)
        exit()

    if args.metrics2:
        metrics2(args.result_folder, args.llm)
        exit()

    print("Loading PDF...")

    loaded: PDFLoader = PDFLoader(args.path)
    print("Loaded PDF:", args.path)

    llm: LLM = LLM()

    template: PromptTemplate = get_extract_templates()[0]
    print("Using template:", template.template)

    revision: str = get_revisions()[0]
    critique: str = get_critiques()[0]

    logfile = "output.log"

    logger.add(logfile, colorize=True, enqueue=True)
    handler = FileCallbackHandler(logfile)

    print("Using revision:", revision)
    print("Using critique:", critique)

    chain = RequirementExtractor(
        llm=llm,
        prompt_template=get_extract_templates()[0],
        principles=[
            ConstitutionalPrinciple(
                critique_request=critique,
                revision_request=revision,
            ),
        ],
        callbacks=[handler],
    ).chain

    pages = get_pages(len(loaded.documents))
    print("Extracting from pages:", pages)

    for page in pages:
        print("Working on page:", page)
        doc: Document = list(
            filter(lambda x: x.metadata["page"] == page - 1, loaded.documents)
        )[
            0
        ]  # zero indexed
        chain.invoke(input=doc)
        print("Done with page:", page)
        print("Saving to output.log")


if __name__ == "__main__":
    main()
