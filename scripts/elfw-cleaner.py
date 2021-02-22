# This code cleans some corrupted hands in the augmented dataset.
# R. Redondo (c) Eurecat 2019

import os

source_path = '../elfw_augmented_facehands'
path_faces = os.path.join(source_path, 'faces')
path_labels = os.path.join(source_path, 'labels')

target_strs = [ 'h2f_web-133',
                'h2f_web-147',
                'h2f_web-157',
                'hof-005',
                'hof-032',
                'hof-042',
                'hof-143',
                'hof-184',
                'Barbara_Boxer',
                'Devin_Harris']

def clean(path):

    for file in os.listdir(path):
        for target in target_strs:
            if target in file:
                origin = os.path.join(path, file)
                destination = os.path.join(source_path, file)
                print('Moving file to ' + destination)
                os.rename(origin, destination)
                break

clean(path_faces)
clean(path_labels)