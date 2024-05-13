import glob
import json
from typing import List
import dataframe_image as dfi
import numpy as np
import pandas as pd
import krippendorff
import os
from datetime import datetime


class ICRProcessor:
    def __init__(self, working_dir, export_dir, iteration: int, llm: str):
        # load json
        self.working_dir = working_dir
        self.dfs_TP_FN = {}
        self.dfs_FP = {}
        self.data_keys = []
        self.export_dir = export_dir
        self.llm = llm
        self.iteration = iteration

    def load_and_process_data_for_icr_TP_FN(self, file_path):
        """Load JSON data from file and process ratings."""
        with open(file_path, "r") as file:
            doc_json = json.load(file)

        data = {"roleplay": [], "fewshot": [], "zeroshot": [], "role_and_few": []}

        for item in doc_json["script"]:
            for key, val in item["ratings"].items():
                for sub_key, sub_val in val.items():
                    data[key].append(
                        {"coder": sub_key, "id": item["id"], "label": sub_val}
                    )
        return data

    def load_and_process_data_for_icr_FP(self, file_path):
        with open(file_path, "r") as file:
            doc_json = json.load(file)

        data = {"roleplay": [], "fewshot": [], "zeroshot": [], "role_and_few": []}

        for key, val in doc_json["FP"].items():
            for page, ratings in val.items():
                for entry in ratings:
                    data[key].append(
                        {
                            "coder": entry["rater"],
                            "number": entry["count"],
                            "page": page,
                        }
                    )
        dfs = {prompt: pd.DataFrame(data[prompt]) for prompt in data}

        return dfs

    def create_and_pivot_dataframe(
        self, data, pivot_index, pivot_columns, pivot_values
    ):
        print(data["fewshot"])
        exit()
        dfs = {}
        for key, values in data.items():
            df = pd.DataFrame(values)

            if not df.empty:  # Check if DataFrame is not empty
                dfs[key] = df.pivot(
                    index=pivot_index, columns=pivot_columns, values=pivot_values
                )

        print(dfs["fewshot"])
        exit()
        return dfs

    def get_majority_labels_fn_tp(self, df):
        arr = df.values
        majority_labels = []
        total_fn = 0
        total_tp = 0
        for column in arr.T:
            tp_count = np.count_nonzero(column == "TP")
            fn_count = np.count_nonzero(column == "FN")
            if tp_count >= fn_count:
                majority_labels.append("TP")
            else:
                majority_labels.append("FN")

        # Display the result
        for label in majority_labels:
            if label == "TP":
                total_tp += 1
            else:
                total_fn += 1

        print(f"total tp: {total_tp}")
        print(f"total fn: {total_fn}")
        return total_fn, total_tp

    def get_average_label_fp(self, df):
        arr = df.values
        total_fp = 0
        for column in arr.T:
            total_fp += np.average(column).round()

        print(f"total fp: {total_fp}")
        return total_fp

    def calculate_krippendorff_one_file(self, file_name: str):
        file_path = os.path.join(self.working_dir, self.llm + f"/{file_name}" ".json")
        print(file_path)
        self.load_and_process_data(file_path)
        # self.export_pngs_for_one_file(file_name)
        self.export_krippendorff_one_file(file_name)

    def load_and_process_data(self, file_path):
        data_TP_FN = self.load_and_process_data_for_icr_TP_FN(file_path)
        data_FP = self.load_and_process_data_for_icr_FP(file_path)
        self.create_dataframes(data_TP_FN, data_FP)

    def create_dataframes(self, data_TP_FN, data_FP):
        self.data_keys = data_TP_FN.keys()
        self.dfs_TP_FN = self.create_and_pivot_dataframe(
            data_TP_FN, "coder", "id", "label"
        )
        self.dfs_FP = self.create_and_pivot_dataframe(
            data_FP, "coder", "page", "number"
        )

    def print_krippendorff_values(self):
        for key in self.data_keys:
            print(f"\n{'='*15} {key}: FN & TP & FP {'='*15}")
            try:
                # TP and FN data using nominal level of measurement
                level_of_measurement = "nominal"
                print("\nTP_FN Data:")
                print(self.dfs_TP_FN[key])
                krippendorff_value = self.calculate_krippendorff_alpha(
                    self.dfs_TP_FN[key], level_of_measurement
                )
                print("#### Krippendorff value #####")
                print(krippendorff_value)

                # FP data using ordinal level of measurement
                level_of_measurement = "ordinal"
                print("\nFP Data:")
                print(self.dfs_FP[key])
                krippendorff_value = self.calculate_krippendorff_alpha(
                    self.dfs_FP[key], level_of_measurement
                )
                print("#### Krippendorff value ####")
                print(krippendorff_value)

                total_fp = self.get_average_label_fp(self.dfs_FP[key])
                total_fn, total_tp = self.get_majority_labels_fn_tp(self.dfs_TP_FN[key])
                r, p, f1 = self.calculate_metrics(total_fn, total_tp, total_fp)

                print("\n")
                print("#### Metrics ####")
                print(f"Recall: {r:.2f}")
                print(f"Precision: {p:.2f}")
                print(f"F1-score: {f1:.2f}")

            except KeyError:
                print(f"\nNo data exists for {key}")

            print("-" * 45)

    def calculate_metrics(self, total_fn, total_tp, total_fp):

        ## recall
        recall = self.calculate_recall(total_fn, total_tp)
        ## precision
        precision = self.calculate_precision(total_tp, total_fp)
        ## f1
        f1 = self.calculate_f1(recall, precision)

        return recall, precision, f1

    def calculate_recall(self, total_fn, total_tp):
        return total_tp / (total_tp + total_fn)

    def calculate_precision(self, total_tp, total_fp):
        return total_tp / (total_tp + total_fp)

    def calculate_f1(self, recall, precision):
        return 2 * ((precision * recall) / (precision + recall))

    def calculate_krippendorff_alpha(self, dataframe, level_of_measurement):
        reliability_data = dataframe.values.tolist()
        data = np.nan
        if len(reliability_data) > 1:
            try:
                data = krippendorff.alpha(
                    reliability_data=reliability_data,
                    level_of_measurement=level_of_measurement,
                )
            except:
                data = 1
        return data

    def export_krippendorff_one_file(self, file_name):
        for key in self.data_keys:

            if key in self.dfs_TP_FN and key in self.dfs_FP:
                file_path_FP = os.path.join(
                    self.export_dir, file_name + f"/{key}/FP/{self.llm}/"
                )
                file_path_TP_FN = os.path.join(
                    self.export_dir, file_name + f"/{key}/TP_FN/{self.llm}/"
                )
                text_file_path_FP = os.path.join(
                    file_path_FP, f"{key}_{self.iteration}_FP.txt"
                )
                text_file_path_TP_FN = os.path.join(
                    file_path_TP_FN, f"{key}_{self.iteration}_TP_FN.txt"
                )

                os.system(f"mkdir -p {file_path_FP} | mkdir -p {file_path_TP_FN}")

                os.system(f"touch {text_file_path_FP} | touch {text_file_path_TP_FN}")

                with open(text_file_path_FP, "w") as fp_file, open(
                    text_file_path_TP_FN, "w"
                ) as tp_fn_file:

                    try:
                        print(
                            f"Saving krippendorff value for {key} as txt to folder {file_path_FP}"
                        )
                        level_of_measurement = "ordinal"
                        krippendorff_value = self.calculate_krippendorff_alpha(
                            self.dfs_FP[key], level_of_measurement
                        )
                        fp_file.write(str(krippendorff_value))

                        level_of_measurement = "nominal"
                        krippendorff_value = self.calculate_krippendorff_alpha(
                            self.dfs_TP_FN[key], level_of_measurement
                        )
                        tp_fn_file.write(str(krippendorff_value))
                    except:
                        print(
                            f"Can't save krippendorffs value, no data exists for {key}"
                        )

    def export_pngs_for_one_file(self, file_name):
        for key in self.data_keys:
            if key in self.dfs_FP and key in self.dfs_TP_FN:

                file_path_FP = os.path.join(
                    self.export_dir, file_name + f"/{key}/FP/{self.llm}/"
                )
                file_path_TP_FN = os.path.join(
                    self.export_dir, file_name + f"/{key}/TP_FN/{self.llm}/"
                )
                print(file_path_FP)
                os.system(f"mkdir -p {file_path_FP} | mkdir -p {file_path_TP_FN}")

                dfi.export(
                    self.dfs_TP_FN[key],
                    file_path_TP_FN + f"{key}_{self.iteration}_TP_FN.png",
                    table_conversion="matplotlib",
                )
                print(f"Saving DataFrames for {key} as png to folder {file_path_TP_FN}")

                dfi.export(
                    self.dfs_FP[key],
                    file_path_FP + f"{key}_{self.iteration}_FP.png",
                    table_conversion="matplotlib",
                )
                print(f"Saving DataFrames for {key} as png to folder {file_path_FP}")

    def main(self, files: list):
        for file_name in files:
            if True:
                print(f"\n################{file_name}#################")
                self.calculate_krippendorff_one_file(file_name)
                self.print_krippendorff_values()
                user_input = input(
                    "Enter any value to continue processing the next file, or type 'exit' to stop: "
                )
                if user_input.lower() == "exit":
                    break
                else:
                    continue
