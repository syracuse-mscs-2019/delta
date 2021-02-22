# ==============================================================================
# This file will process the wave files from our dataset which can be found 
# at the following link.
#
# https://www.kaggle.com/mfekadu/english-multispeaker-corpus-for-voice-cloning
# 
# It will block the wave files into fixed length blocks and then extract the
# filter bank (fbank) features using the DELTA library. 
#
# author: Matthew Kinsley
# ==============================================================================
import os
import numpy as np
import pandas as pd
import wave

from pathlib import Path
from delta.data.feat import speech_feature

# ==============================================================================
# Constants that define features that can be modified by users to change 
# what is created by the system.
# ==============================================================================
DATA_PATH = '../../data/wav48'      # Loction data is stored
WORK_PATH = '../../data/working'    # Working folder to use for blocks.
FEAT_PATH = './fbanks'              # folder to output the feature files to.
FILE_LEN = 60                       # Length is in seconds
TRAIN_SUBJ = 78                     # Number of subjects for the training set
BLOCKS = 20                         # Number of blocks for each subject
FEAT_CONFIG = {                     # Config object for buildng features
    'sample_rate': 48000,
    'window_length': 0.025,
    'window_step': 0.010,
    'feature_size': 40
}

if os.path.exists(WORK_PATH) is False:
    os.makedirs(WORK_PATH)          # Create paths that do not exist
if os.path.exists(FEAT_PATH) is False:
    os.makedirs(FEAT_PATH)          # Create paths that do not exist

# ==============================================================================
# Description:
#    This function finds the next file in the folder that hasn't been processed
# yet.
#
# Inputs:
#    data_path - The path where files are stored
#    idx - The index of the suspected next file
#
# Output:
#    is_good - True if a file was found, otherwise False
#    file_path - The full file name and path string.
#    idx - The suspected idx of the next file
# ==============================================================================
def get_next_file_name(data_path, idx):
    is_good = False
    file_path = data_path + '_' + str(idx).zfill(3) + '.wav'

    # Give it 3 attempts to find a good file because some file ids are missing
    x = 0
    while not Path(file_path).exists() and x < 3:
        idx = idx+1
        file_path = data_path + '_' + str(idx).zfill(3) + '.wav'
        x = x+1

    # If we found a file make sure is_good is true
    if Path(file_path).exists():
        is_good = True

    return is_good, file_path, idx+1


# ==============================================================================
# Description:
#    This function loads the original waveforms and then blocks them into the  
# specified number of blocks eccach of a specified length.
#
# Inputs:
#    data_path - The path where files are stored
#    working_path - The path were the working files are created.
#    blocks - The number of files to create for each subject.
#    len - The number of seconds each files should be.
#
# Output:
#    DataFrame containing the subject ID and file name for each file created.
# ==============================================================================
def prep_wave_files(data_path, working_path, blocks, len_sec):
    files = pd.DataFrame({}, columns=['Subject','FileName'])

    # Get a listing of all dir entries in the data_path    
    entries = os.scandir(data_path)

    # Each directory represents a subject.  We need to load those files an build
    # the necessary amount of 60 second files.  
    for entry in entries:
        subject_id = entry.name
        path = entry.path

        print('Processing Subject ', subject_id)

        j = 1
        bcnt = 0
        audiof = []
        done = False
        while not done and bcnt < blocks:
            out_file_name = working_path + '/' + subject_id + '_' + str(bcnt).zfill(3) + '.wav'
            is_good, in_file_name, j = get_next_file_name(path + '/' + subject_id, j)

            # If good then read in the wave file
            if is_good:
                # read in the file
                wf = wave.open(in_file_name, 'rb')

                # Append it to the working stream
                audiof.append([wf.getparams(), wf.readframes(wf.getnframes())])
                sr = wf.getparams().framerate
                wf.close()

                # Check to see if we have exceeded 60 seconds yet
                length = 0
                for i in range(0,len(audiof)):
                    length = length + audiof[i][0].nframes

                if length > len_sec*sr:
                    print('    writing block: ', bcnt)

                    # Calculate important parameters for writing the file
                    l_block = len(audiof)-1
                    l_block_overlow = (length-len_sec*sr)
                    l_block_len = (audiof[l_block][0].nframes-l_block_overlow) * audiof[l_block][0].sampwidth

                    # Write the block
                    output = wave.open(out_file_name, 'wb')
                    output.setparams(audiof[0][0])
                    for i in range(0, len(audiof)-1):
                        output.writeframes(audiof[i][1])
                    
                    output.writeframes(audiof[l_block][1][0:l_block_len])
                    output.close()

                    # Increment the block count
                    # Save the remainder
                    naudiof = []
                    naudiof.append([audiof[l_block][0], audiof[l_block][1][l_block_len:]])
                    audiof = naudiof
                    bcnt = bcnt + 1

                    # Append the file to the list of files created
                    obj = pd.DataFrame({'Subject': [subject_id], 'FileName': [out_file_name]})
                    files = pd.concat([files, obj], ignore_index=True, axis=0)
            else:
                done = True

        print('Subject ', subject_id, ' Complete')

    return files

# ==============================================================================
# Description:
#    This function process a group of files creating the fbank and numpy files
# in the specified directory.
#
# Inputs:
#    data_path - The path where files are stored
#    output_path - The path were the feature files are to be stored.
#    df_files - The data frame containing the files and other necessary
#               information.
#
# Output:
#    A dataframe that details the subjects and location of the feature files.
# ==============================================================================
def create_feature_files(data_path, output_path, df_files, feat_config):
    files = df_files.copy()

    # Extract the feature settings
    wlen = feat_config['window_length']
    wstep = feat_config['window_step']
    srate = feat_config['sample_rate']
    fsize = feat_config['feature_size']

    # load all the files needer per subjects
    l = df_files.shape[0]
    for i in range(0,l):
        print('Processing file ', i+1, ' of ', l)
        file_name = df_files.at[i, 'FileName']
        feat_file = savepath = os.path.splitext(file_name)[0] + '.npy'

        # DELTA function that extracts from the file the features
        # the resulting data gets dumped to an equivalently named
        # feature numpy file.
        speech_feature.extract_feature(file_name,
                        winlen=wlen,
                        winstep=wstep,
                        sr=srate,
                        feature_size=fsize,
                        feature_name='fbank')

        # Move the file to the feature path
        dest_file = output_path + '/' + os.path.basename(feat_file)
        os.rename(feat_file, dest_file)

        # Replace the file name with the feature file name
        files.at[i, 'FileName'] = dest_file
    return files


# ==============================================================================
# Description:
#    This function takes a data frame and then mixes it up with all possible
# combinations.
#
# Inputs:
#    df_data - The dataframe to expand with all combinations.
#
# Output:
#    df_exp - The expanded dataframe
# ==============================================================================
def expand_data_set(df_data):
    df_exp = pd.DataFrame({}, columns=['Match','FileName1', 'FileName2'])
    l = df_data.shape[0]
    for i in range(0, l):
        print('     Processing Row',i)
        subj = df_data.at[i, 'Subject']
        df_sub = pd.DataFrame({}, columns=['Match','FileName1', 'FileName2'])
        for j in range(0, l):
            if i != j:
                obj = pd.DataFrame({
                    'Match': [df_data.at[i, 'Subject'] == df_data.at[j, 'Subject']], 
                    'FileName1': [df_data.at[i, 'FileName']],
                    'FileName2': [df_data.at[j, 'FileName']]
                })
                df_sub = pd.concat([df_sub, obj], ignore_index=True, axis=0)
        df_exp = pd.concat([df_exp, df_sub], ignore_index=True, axis=0)

    return df_exp


# ==============================================================================
# Description:
#    This function creates the training and test set files in the agreed to
# data format described here.
#
# Inputs:
#    data_path - The path where files are stored
#    df_files - The data frame containing the files and other necessary
#               information.
#
# Output:
#    None
# ==============================================================================
def create_train_test_files(data_path, df_files, train_subj):
    # Get the unique subject values
    nsubj = df_files['Subject'].nunique()
    subjs = df_files['Subject'].unique()

    # Initialize the dataframes
    df_train = pd.DataFrame({}, columns=['Subject','FileName'])
    df_test = pd.DataFrame({}, columns=['Subject','FileName'])

    # Generate the output files
    print('Splitting the dataset')
    for i in range(0, nsubj):
        s = subjs[i]
        f = df_files[df_files['Subject']==s]
        if i < train_subj:
            # Get all subjects matching this ID.
            df_train = pd.concat([df_train, f], ignore_index=True, axis=0)
        else:
            # Output to the test set
            df_test = pd.concat([df_test, f], ignore_index=True, axis=0)
    
    # Next we have to create the expanded files
    print('Expanding the training set')
    df_train_exp = expand_data_set(df_train)

    print('Expanding the training set')
    df_test_exp = expand_data_set(df_test)

    # Write the files to a file
    df_train.to_csv('./trainset.csv')
    df_train_exp.to_csv('./trainset_opt2.csv')
    df_test_exp.to_csv('./testset.csv')

    return

# ==============================================================================
# The ocde here calls the helper functions to generate the test and train files.
# ==============================================================================
df_data = prep_wave_files(DATA_PATH, WORK_PATH, BLOCKS, FILE_LEN)
df_data = create_feature_files(WORK_PATH, FEAT_PATH, df_data, FEAT_CONFIG)
create_train_test_files(FEAT_PATH, df_data, TRAIN_SUBJ)