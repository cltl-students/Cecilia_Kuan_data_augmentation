"""
Check data file for missing values.

"""

import sys
import csv
import glob
from pathlib import Path

def check_persona_data(persona_file, data_file):
    missing = []
    found_dict = {}

    with open(persona_file) as file:
        personas = list(csv.reader(file, delimiter=";"))
        print(">>>> Number of Personas", len(personas), "==")

        with open(data_file) as ofile:
            results = list(csv.reader(ofile, delimiter="\t"))
            # filter out "output for:" line
            filtered = list(filter(lambda item: len(item[1]) > 0, results))

            for idx in range(len(personas)):
                gender, age, job, illness = personas[idx]
                gender = gender.lower()
                age = age.lower()
                job = job.lower()
                illness = illness.lower()

                for item in filtered:
                    try:
                        _idx, _note, _score, _cat, _level, _job, _gender, _age, _illness = item[0:9]
                        _job = _job.lower()
                        _gender = _gender.lower()
                        _age = _age.lower()
                        _illness = _illness.lower()
                        if gender == _gender and age == _age and job == _job and illness == _illness:
                            found_dict[str(idx)] = True
                    except ValueError as e:
                        print("== ERR:", item)
                        raise e

        found_dict = [int(x) for x in list(found_dict.keys())]
        if len(found_dict) != len(personas):
            for x in range(len(personas)):
                try:
                    idx = found_dict.index(x)
                except ValueError:
                    missing.append(personas[x])

    return missing

#
# MAIN
missing_persona_data = []
if len(sys.argv) != 3:
    print("USAGE: python cece.py PersonaDirectory DataDirectory")
    exit(255)

persona_dir = sys.argv[1]
data_dir = sys.argv[2]

if persona_dir[-1] != "/":
    persona_dir += "/"
if data_dir[-1] != "/":
    data_dir += "/"

persona_files = glob.glob(f"{persona_dir}persona_*.csv")
# print(persona_files)

for persona_file in persona_files:
    bname = Path(persona_file).stem
    parts = bname.split("_")
    parts.pop(0)
    persona_name = "_".join(parts)
    # _, persona_name = bname.split("_")

    data_files = glob.glob(f"{data_dir}data_*_{persona_name}.tsv")
    # print(data_files)
    for data_file in data_files:
        print(f"== PROCESSING {persona_file} and {data_file} ==")
        bname = Path(data_file).stem
        parts = bname.split("_")
        parts.pop(0) # remove data
        cat = parts.pop(0) # get b140
        level = parts.pop(0) # get level
        name = "_".join(parts)
        #_, cat, level, name = bname.split("_")
        missing = check_persona_data(persona_file, data_file)
        if len(missing) > 0:
            print(f">>>> OOPS: MISSING {len(missing)} data for persona")
            for item in missing:
                gender, age, job, illness = item
                missing_persona_data.append(','.join([
                    name,
                    cat,
                    level,
                    gender,
                    age,
                    job,
                    illness,
                ]))

print("\n=================================================================\n")
if len(missing_persona_data) == 0:
    print("== DATA IS ALL GOOD!!!")
else:
    print(f"== MISSING {len(missing_persona_data)} persona data:\n")
    for item in missing_persona_data:
        print(item)
print("\n=================================================================\n")
