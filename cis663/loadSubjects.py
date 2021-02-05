# ==============================================================================
# This file defines a class that contains a method that takes in a 
# file name and then using delta loads the wave files for that subject.
#
# Based on a configuration dictionary that is provided to the method it
# will segregate the files up into a train and test set.
#
# author: Matthew Kinsley
# ==============================================================================
import pandas as pd
import delta.compat as tf
from pathlib import Path
from delta.data.frontend.read_wav import ReadWav


# ==============================================================================
# Description:
#   This method takes in a file name and a set of configuration data and loads
#   the wav file.
#
# author: Matthew Kinsley
#
# inputs:
#    fileName - The file name of the wav file to load.
#    func_config - A dictonary containing settings.
#
# output: 
#    The loaded wav object  
# ==============================================================================
def LoadWavFile(fileName, func_config):
    config = {'sample_rate': func_config['sample_rate']}
    read_wav = ReadWav.params(config).instantiate()
    data, srate = read_wav(fileName)
    return data

# ==============================================================================
# This method loads wave files into a pandas array.
#
# Based on a configuration dictionary that is provided to the method it
# will segregate the files up into a train and test set.
#
# author: Matthew Kinsley
# ==============================================================================
# ==============================================================================
# Description:
#   This file loads the subject numbers from a file and then load all of 
#   the requsted files for each of the listed subjects into a DataFrame.
#
# author: Matthew Kinsley
#
# inputs:
#    fileName - The file name of the wav file to load.
#    func_config - A dictonary containing settings.
#
# output: 
#    A data frame containing the subject ID, sample rate of the file and the
#    wav file loaded. 
# ==============================================================================
def LoadSubjects(fileName, func_config):
    dfData = pd.DataFrame({}, columns=['Subject', 'sample_rate', 'wavFile'])

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
            
            x = 0
            while not Path(fName).exists() and x < 3:
                fidx = fidx+1
                fName = data_path + '/' + d.loc[i] + '/' + d.loc[i] + '_' + str(fidx).zfill(3) + '.wav'
                x = x+1

            if Path(fName).exists():
                data = LoadWavFile(sess, fName, func_config)

                # add the data and srate touple to a list
                obj = {
                    'Subject': d.loc[i], 
                    'sample_rate': srate,
                    'wavFile': data
                }
                dfData = dfData.append(obj, ignore_index=True)
            
            fidx = fidx+1

    return dfData
