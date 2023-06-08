import FaceRecognition
import numpy as np
import os

path_dataset = os.path.join(os.getcwd(),'trial_lfw')
print(path_dataset) #for checking

FR = FaceRecognition.Face_Recognition(path_dataset)
print(FR.number_images())
print(FR.number_identities())
total_names, total_data = FR.data_images()
print(len(total_names))

total_data = np.array(total_data)
print(np.shape(total_data))
print(type(total_data[0]))
print(total_names)