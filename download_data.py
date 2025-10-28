"""
This script extracts text from  downloaded json datasets
Ignores all metadata, images etc. about articles and
saves the texts in their own monthly files.
"""

import os
import json


# Save all text to files so it is easier to process later
output_dir = "project/extracted_texts"
os.makedirs(output_dir, exist_ok=True)

def get_yearly_texts(data_dir, year):

    #make new dir for every year:
    os.makedirs(os.path.join(output_dir, str(year)), exist_ok=True)

    #process all subdirs for every month:
    for month_folder in sorted(os.listdir(data_dir)):
        month_path = os.path.join(data_dir, month_folder)
        if os.path.isdir(month_path):
            month_texts = []
            for file_name in os.listdir(month_path):
                if file_name.endswith(".json"):
                    print(file_name)
                    file_path = os.path.join(month_path, file_name)

                    #open the json file:
                    with open(file_path, "r", encoding="utf-8") as f:
                        try:
                            data = json.load(f)

                            #find the text from the json structure:
                            if "data" in data:
                                for article in data["data"]:
                                    if "content" in article:
                                        for block in article["content"]:
                                            text = block.get("text")
                                            if text:
                                                if isinstance(text, list):
                                                    text = " ".join(map(str, text))
                                                else:
                                                    text = str(text)
                                                month_texts.append(text)

                        except json.JSONDecodeError:
                            print(f"Warning: Could not parse {file_path}")
            

            # join all monthly text into one large string
            all_text = "\n".join(month_texts)

            # save every month to own txt file (inside year dir)
            output_path = os.path.join(os.path.join(output_dir, str(year)), f"{month_folder}.txt")
            with open(output_path, "w", encoding="utf-8") as out_f:
                out_f.write(all_text)


# For year 2021:
data_dir_2021 = "project/ylenews-fi-2021-src/2021"
get_yearly_texts(data_dir_2021, 2021)


# For years 2019-2020:
data_dir_2019 = "project/ylenews-fi-2019-2020-src/2019"
get_yearly_texts(data_dir_2019, 2019)
data_dir_2020 = "project/ylenews-fi-2019-2020-src/2020"
get_yearly_texts(data_dir_2020, 2020)


# For years 2022-2024:
for i in range(3):
    year = 2022 + i
    print(year)
    data_dir = "project/ylenews-fi-2022-2024-src/" + str(year)
    get_yearly_texts(data_dir, year)


# For years 2011-2018:
for i in range(8):
    year = 2011 + i
    print(year)
    data_dir = "project/ylenews-fi-2011-2018-src/data/fi/" + str(year)
    get_yearly_texts(data_dir, year)



