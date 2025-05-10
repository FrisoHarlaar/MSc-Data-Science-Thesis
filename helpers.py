import pandas as pd
import numpy as np
import json
import gzip
from tqdm.notebook import tqdm
import re 
from sklearn.model_selection import train_test_split
import warnings

from emotions import emotion_to_int


def read_goodreads_data(file_path, max_rows=None, sample_size=10000, return_sample=True):
    """
    Read Goodreads JSON.GZ data into a DataFrame
    
    Parameters:
    -----------
    file_path : str
        Path to the goodreads_books.json.gz file
    max_rows : int, optional
        Maximum number of rows to read (None = read all)
    sample_size : int, optional
        Number of rows to sample if return_sample=True
    return_sample : bool, default=True
        If True, return a random sample instead of the full dataset
        
    Returns:
    --------
    DataFrame containing book data
    """
    all_books = []
    total_processed = 0
    
    # For sampling
    if return_sample:
        # First pass to count total lines (if we need exact sampling)
        if not max_rows:
            print("Counting total records for sampling...")
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                total_lines = sum(1 for _ in tqdm(f))
            sampling_rate = min(1.0, sample_size / total_lines)
            print(f"Sampling rate: {sampling_rate:.4f} ({sample_size} of {total_lines:,})")
        else:
            # If max_rows is specified, use that for sampling rate calculation
            total_lines = max_rows
            sampling_rate = min(1.0, sample_size / max_rows)
    
    # Read the file
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for i, line in tqdm(enumerate(f)):
            # Stop if we reached max_rows
            if max_rows and i >= max_rows:
                break
                
            # Sample if requested
            if return_sample and np.random.random() > sampling_rate:
                continue
                
            try:
                # Parse JSON line and append to list
                book = json.loads(line.strip())
                all_books.append(book)
                total_processed += 1
                
                # Print progress for large datasets
                if total_processed % 100000 == 0 and not return_sample:
                    print(f"Processed {total_processed:,} records")
                    
            except json.JSONDecodeError:
                print(f"Error parsing JSON at line {i}")
    
    print(f"Creating DataFrame with {len(all_books):,} records...")
    df = pd.DataFrame(all_books)
    
    return df


def slugify(text):
    """
    Convert string to lowercase, replace spaces and invalid chars with hyphens.
    Handles potential None input.
    """
    if text is None:
        return ""
    text = str(text).lower()
    # Remove characters that are definitely not allowed or likely not used
    text = re.sub(r'[^\w\s-]', '', text)
    # Replace whitespace with hyphens
    text = re.sub(r'\s+', '-', text)
    # Remove leading/trailing hyphens
    text = text.strip('-')
    return text


def make_train_test_val_splits(dataset_df, loads, random_seed, unique_id_column=None):
    """ Split the data into train/val/test.
    :param dataset_df: pandas Dataframe containing the dataset (e.g., ArtEmis)
    :param loads: list with the three floats summing to one for train/val/test
    :param random_seed: int
    :return: changes the dataset_df in-place to include a column ("split") indicating the split of each row
    """
    if sum(loads) != 1:
        raise ValueError()

    train_size, val_size, test_size = loads
    print("Using a {},{},{} for train/val/test purposes".format(train_size, val_size, test_size))
    
    test_num = int((test_size+val_size) * len(dataset_df[dataset_df.version == 'new']))
    dataset_df['unique_id'] = dataset_df['art_style']+dataset_df['painting']
    gdf = dataset_df.groupby(by=['unique_id', 'version']).agg(
        {
            'repetition': 'first',
        }
    ).reset_index()
    start_rep = 5
    test_ids = []
    while (test_num > 0) and (start_rep < 10):
        gdf_rep = gdf[(gdf.version=='new') & (gdf.repetition==start_rep)]
        curr_cnt = len(gdf_rep) * start_rep
        if test_num > curr_cnt:
            test_ids.extend(gdf_rep.unique_id.values.flatten())
            test_num -= curr_cnt
            start_rep += 1
        else:
            test_ids.extend(gdf_rep.unique_id.values.flatten()[:int(test_num//start_rep)])
            test_num -= curr_cnt
            start_rep += 1
    print(f'New test set has maximum repetition {start_rep-1}')

    new_set_unique_ids = set(gdf[gdf.version=='new'].unique_id.values)
    new_set_train_ids = np.array(list(new_set_unique_ids - set(test_ids)))
    test_end = int(len(test_ids)*(test_size/(test_size+val_size)))
    new_set_test_ids = np.array(test_ids)[:test_end]
    new_set_val_ids = np.array(test_ids)[test_end:]

    df = dataset_df
    ## unique id
    # if unique_id_column is None:
    unique_id = df.unique_id # default for ArtEmis
    anchor_id = df.anchor_art_style + df.anchor_painting
    anchor_id = anchor_id.dropna()
    # else:
    #     unique_id = df[unique_id_column]

    unique_ids = set(unique_id.unique())
    # unique_ids.sort()
    unique_ids -= new_set_unique_ids

    anchor_ids = set(anchor_id.unique())
    # anchor_ids.sort()
    anchor_ids -= new_set_unique_ids

    train_len = len(unique_ids) * train_size
    test_len = len(unique_ids) * (test_size + val_size)

    rem_train = train_len - len(anchor_ids)

    # assert rem_train < 0, 'unique anchors is more than remaining images'
    while rem_train < 0:
        warnings.warn('Anchor paintings are more than remaining paintings .... Removing some anchor painting')
        anchor_ids.pop()
        rem_train += 1
    # unique_ids_rem = np.array([i for i in unique_ids if i not in anchor_ids])
    unique_ids_rem = np.array(list(unique_ids - anchor_ids))

    train, rest = train_test_split(unique_ids_rem, test_size=(test_len)/(test_len+rem_train), random_state=random_seed)

    train = set(np.concatenate((train, list(anchor_ids), new_set_train_ids)))

    if val_size != 0:
        val, test = train_test_split(rest, test_size=test_size/(test_size+val_size), random_state=random_seed)
    else:
        test = rest
    test = set(np.concatenate((test, new_set_test_ids)))
    assert len(test.intersection(train)) == 0

    def mark_example(x):
        if x in train:
            return 'train'
        elif x in test:
            return 'test'
        else:
            return 'val'

    df = df.assign(split=unique_id.apply(mark_example))
    df.drop(columns=['anchor_art_style', 'anchor_painting', 'unique_id'], inplace=True)
    return df


def preprocess(path, verbose=True):
    """ Split data, drop too short/long, spell-check, make a vocabulary etc.
    """

    #1. load the provided raw ArtEmis csv
    df = pd.read_csv(path)
    if verbose:
        print('{} annotations were loaded'.format(len(df)))

    #3. split the data in train/val/test  (the splits are based on the unique combinations of (art_work, painting).
    df = make_train_test_val_splits(df, [0.85, 0.05, 0.1], 6552)

    #8. encode feelings as ints
    df['emotion_label'] = df.emotion.apply(emotion_to_int)
    return df