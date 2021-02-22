# ==============================================================================
# This file is a starting point for extracting features from our raw data 
# files. 
#
# author: Matthew Kinsley
# ==============================================================================
import pandas as pd
import delta.compat as tf
import numpy as np
from pathlib import Path
from delta.data.frontend.read_wav import ReadWav
from loadSubjects import LoadSubjects
from delta.data.frontend.mel_spectrum import MelSpectrum
from delta.data.feat import speech_feature


# ==============================================================================
# Description:
#   This method takes the a DataFrame with wav file data and replaces the
#   wav file data with the mel spectrum data.
#
# author: Matthew Kinsley
#
# inputs:
#    data - Is the DataFrame containing the wave files to process
#    func_config - Is a dictionary with inputs used by this function.
#
# output: 
#    a new data frame where the wave object is replaced by the melspectrum data.
# ==============================================================================
def GetMelSpecFeatures(data, func_config):
    sample_rate = func_config['sample_rate']

    config = {
        'window_type': 'hann',
        'upper_frequency_limit': 7600,
        'filterbank_channel_count': 80,
        'lower_frequency_limit': 80,
        'dither': 0.0,
        'window_length': 0.025,
        'frame_length': 0.010,
        'remove_dc_offset': False,
        'output_type': 3,
        'sample_rate': 48000,
        'snip_edges': True,
        'raw_energy': 1
    }

    df_res = pd.DataFrame({})
    s = dfData.shape
    for i in range(0, s[0]):
        mel_spectrum = MelSpectrum.params(config).instantiate()
        mel_spectrum_test = mel_spectrum(data.at[i, 'wavFile'], sample_rate)
        mspec = mel_spectrum_test.eval(session=sess)
        mspec = mspec.flatten()
        df_res = df_res.append(pd.Series(mspec), ignore_index=True)

    return pd.concat([dfData['Subject'], df_res], axis=1, ignore_index=True)


# ==============================================================================
# Description:
#   This method takes in a file name containing the subject IDs of the subjects 
#   to load and a dictionary with settings relevent to the feature extraction
#   process.
#
# author: Matthew Kinsley
#
# inputs:
#    fileName - The file name containing the subject IDs to load.
#    func_config - A dictonary containing settings needed to load the data.
#
# output: 
#    TBD: Nothing right now - I need to figure out how the models accept it as
#    input.  Looks like the npy file is a matrix that roughly represents an image
#    so the next question is how should data be configured.  
# ==============================================================================
def getFeatures(fileName, func_config):
    wlen = func_config['window_length']
    wstep = func_config['window_step']
    srate = func_config['sr']
    fsize = func_config['feature_size']
    feat = func_config['features']

    # Get the subject IDs form the files
    subs = pd.read_csv(fileName)    
    d = subs['SubjectID']
    l = d.size

    # For each subject load the necessary number of files an add them to the
    # Train and test sets.
    srate = func_config['sample_rate']
    data_path = func_config['data_path']
    num_files_per = func_config['files_per_subj']

    # load all the files needer per subjects
    wavFiles = []
    for i in range(0,l):
        fidx = 0
        for j in range(1, num_files_per+1):
            fName = data_path + '/' + d.loc[i] + '/' + d.loc[i] + '_' + str(fidx).zfill(3) + '.wav'
            featFile = data_path + '/' + d.loc[i] + '/' + d.loc[i] + '_' + str(fidx).zfill(3) + '.npy'
            
            x = 0
            while not Path(fName).exists() and x < 3:
                fidx = fidx+1
                fName = data_path + '/' + d.loc[i] + '/' + d.loc[i] + '_' + str(fidx).zfill(3) + '.wav'
                featFile = data_path + '/' + d.loc[i] + '/' + d.loc[i] + '_' + str(fidx).zfill(3) + '.npy'
                x = x+1

            if Path(fName).exists():
                if(feat == 1):
                    speech_feature.extract_feature(fName, # delta method to get numerics from wav files
                                    winlen=wlen,
                                    winstep=wstep,
                                    sr=srate,
                                    feature_size=fsize,
                                    feature_name='fbank')
                else:
                    speech_feature.extract_feature(fName, # same as above, but much larger data
                                    winlen=wlen,
                                    winstep=wstep,
                                    sr=srate,
                                    feature_size=fsize,
                                    feature_name='spec')
            
            feat = np.load(featFile)
            print(feat.shape)
            fidx = fidx+1


# ==============================================================================
# Test stub code.
# ==============================================================================
# Constants for the script
FILE_NAME = './subjects.csv'
MY_CONFIG = {
    'sample_rate': 48000,
    'data_path': '../../data/wav48',
    'files_per_subj': 1,
    'window_length': 0.025,
    'window_step': 0.010,
    'sr': 48000,
    'feature_size': 40,
    'features': 1
}

# start of the script, and we need to have an active 
# session.  Not sure how they create the one they use
# but this is a default one and seems to allow things
# to work.
sess = tf.compat.v1.Session()

try:
    # Code to load the Mel Spectrum data
    #dfData = LoadSubjects(FILE_NAME, MY_CONFIG)
    #print("dfData shape:", dfData.shape)

    #dfFeat = GetMelSpecFeatures(sess, dfData, MY_CONFIG)
    #print("dfFeat shape:", dfFeat.shape)
    #dfFeat.to_csv('./melspec_features.csv')

    # Load the fbank features, right now it kind of just
    # converts the files to npy files containing the
    # extrated features.
    getFeatures(FILE_NAME, MY_CONFIG)
finally:
    sess.close()