from ICR_processor import ICRProcessor
import argparse
import sys

trial = "240429"

if __name__ == "__main__":

    STANDARDS = ["SS_EN_60871_1_EN", "ISO_8528_8_2016_EN", "IEC_IEEE_60214"]
    working_dir = f"input_output/trial_{trial}/"
    export_dir = f"input_output/trial_{trial}/IRR_output/"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        help="please provide the model to be used in the IRR, either: 'bing' or 'local'",
        required=True,
    )
    args = parser.parse_args()
    llm: str
    match args.model:
        case "local":
            llm = "llm_rated"
            print("RUNNING LOCAL LLM DATA ..............................\n")
        case "bing":
            llm = "bing_rated"
            print("RUNNING BING LLM DATA ..............................\n")
        case _:
            print(
                "--model needs to be either bing or local, example usage --model=local"
            )
            sys.exit(1)

    icrp_test = ICRProcessor(
        working_dir=working_dir, export_dir=export_dir, iteration=trial, llm=llm
    )

    icrp_test.main(STANDARDS)
