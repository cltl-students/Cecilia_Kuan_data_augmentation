"""
code adapted and modified from A-PROOF repository https://github.com/cltl/a-proof-zonmw
==============================

:::::: GPT data ::::::

Process a batch of annotated tsv files from GPT data generation output
Save the resulting DataFrame in pkl format.

Process data:
- change ICF category labels to match project acronyms, labels=['ADM', 'ATT', 'BER', 'ENR', 'ETN', 'FAC', 'INS', 'MBW', 'STM']
- add column "labels" - list of binary showing the category
- lower-case all columns except notes
- add column name "text" for notes

Customizable parameters:
    --batch_dir: path to the directory with the annotated batch files
    --outfile: filename for the output pkl
    --update_annotated: if True, update the register of annotated notes ID's with the notes from df
    --update: if passed, the register of annotated notes is not updated with the current note ID's
    --annotfile: path to the register of already annotated notes ID's
"""
batch_dir, outfile, update_annotated, annotfile, outfile_tsv

import pandas as pd
import glob
import csv
import sys
sys.path.insert(0, '../..')


def filename_parser(tsv):
    """
    Parse a filename and return a dict with identifying metadata.

    This parser is used for the naming convention implemented in the a-proof-zonmw project.
    Filename convention:
    'institution--year--MDN--NotitieID--batch.conll'
    Example:
    'vumc--2020--1234567--123456789--batch3.conll'

    # b140_0_nurse.tsv
    # cat_lvl_annotator.tsv

    Parameters
    ----------
    tsv: Path
        path to tsv file (GPT output)

    Returns
    -------
    dict
        dictionary of metadata
    """
    annotator = tsv.stem #returns the full file name without extension
    conll = tsv.parent
    institution, year, MDN, NotitieID, batch = conll.stem.split('--')

    return dict(
        annotator = annotator.lower(),
        institution = institution.lower(),
        year = year,
        MDN = MDN,
        NotitieID = NotitieID,
        batch = batch,
        legacy_rawfile = None,
    )


def tsv_to_df(filepath, filename_parser=filename_parser):
    """
    Parse an GPT output file (tsv format) into pd DataFrame.

    Parameters
    ----------
    filepath: Path
        path to tsv file (GPT output)
    filename_parser:
        filename_parser function to use for metadata extraction

    Returns
    -------
    DataFrame
        dataframe of annotations from tsv and metadata from filename
    """

    names = ['id', 'text', 'conf_score', 'labels', 'level', 'annotator', 'gender', 'age', 'illness']
    cat_list = ['b440','b140', 'd840-d859','b1300','d550','d450','b455','b530','b152']

    with open(filepath) as ofile:
        results = list(csv.reader(ofile, delimiter="\t"))
        results = list(filter(lambda item: not item[0].startswith("output for"), results))

        # results = [item[0:9] for item in results]
        results = [[item.lower().strip() for item in row[0:9]] for row in results]
        for row in results:
            matches = list(set(cat_list).intersection(set([row[3]])))
            if len(matches) == 0:
            #if row[3] not in cat_list:
                print("===== PROBLEM:", row, filepath)

        df = pd.DataFrame(results, columns = names)
    return df


def tsv_to_df_backup(filepath, filename_parser=filename_parser):
    """
    Parse an GPT output file (tsv format) into pd DataFrame.

    Parameters
    ----------
    filepath: Path
        path to tsv file (GPT output)
    filename_parser:
        filename_parser function to use for metadata extraction

    Returns
    -------
    DataFrame
        dataframe of annotations from tsv and metadata from filename
    """
    #metadata = filename_parser(filepath)

    try:
        names = ['id', 'text', 'conf_score', 'labels', 'level', 'annotator', 'gender', 'age', 'illness']
        df = pd.read_csv(
            filepath,
            sep='\t',
            quoting=3,
            names=names,
            encoding='utf-8',
            low_memory=False
        ).dropna(how='all')#.query("sen_tok.str[0] != '#'").assign(**metadata)

        df['file_name'] = filepath
        discard = ["output"]
        df = df[~df.id.str.contains('|'.join(discard), na=False)]
        df['labels'].apply(lambda x: x.strip())

        return df
    except (pd.errors.ParserError, AttributeError) as e:
        print("=============== ERR:", filepath)
        raise e

    #read_csv (skiprows=5)

def update_annotated_notes_ids(df, fp):
    """
    Update the register of annotated notes ID's with the notes from df.
    If register does not exist, create it.

    Parameters
    ----------
    df: DataFrame
        dataframe of annotations (batch)
    fp: Path
        path to the file that logs annotated notes

    Returns
    -------
    None
    """
    cols = ['institution', 'year', 'MDN', 'NotitieID', 'batch', 'legacy_rawfile']
    annotated_notes_ids = df[cols].drop_duplicates()
    print(f"Number of unique annotated notes in this batch: {annotated_notes_ids.shape[0]}")

    if fp.exists():
        existing_notes_ids = pd.read_csv(fp)
        print(f"Number of unique notes from previous annotations: {existing_notes_ids.shape[0]}")
        annotated_notes_ids = existing_notes_ids.append(annotated_notes_ids
        ).drop_duplicates(subset='NotitieID')

    annotated_notes_ids.to_csv(fp, index=False)
    print(f"Total number annotated notes: {annotated_notes_ids.shape[0]}")
    print(f"Updated file saved to: {fp}")


def main(batch_dir, outfile, update_annotated, annotfile, outfile_tsv): #legacy_parser=None, path_to_raw=None):
    """
    Process a batch of annotated tsv files (GPT output).
    Save the resulting DataFrame in pkl format.
    Update the register of annotated notes ID's so that the notes are excluded from future selections.

    Parameters
    ----------
    batch_dir: str
        path to a batch of annotation outputs
    outfile: str
        filename for the output pkl
    update_annotated: bool
        if True, add the notes ID's of current annotations to `annotfile`
    annotfile: str
        path to the register of already annotated notes ID's
    outfile_tsv: str
        path and filename of .tsv to save processed data

    Returns
    -------
    None
    """

    # select filename parser
    global filename_parser

    # process tsv files in all subdirectories of batch_dir
    print(f"...... Processing tsv files in {batch_dir} ......")
    annotated = pd.concat((tsv_to_df(fp, filename_parser) for fp in glob.glob(batch_dir+'*.tsv')), ignore_index=True)
    print(f">>>>>> DataFrame created: {annotated.shape=} <<<<<<")

    annotated.to_pickle(outfile)
    print(f">>>>>> DataFrame saved to {outfile} <<<<<<")

    annotated.to_csv(outfile_tsv, sep="\t")
    print(f">>>>>> DataFrame saved to {outfile_tsv} <<<<<<")

    # save the id's of the annotated notes
    if update_annotated:
        annotfile = Path(annotfile)
        update_annotated_notes_ids(annotated, annotfile)


if __name__ == '__main__':

    dir = '../../data/'
    batch_dir = dir+'tsv/'
    outfile = dir+'gpt_data.pkl'
    annotfile = dir+'annotfile'
    outfile_tsv = dir+'gpt_data.tsv'


    main(
        batch_dir,
        outfile,
        False,
        annotfile,
        outfile_tsv,
    )