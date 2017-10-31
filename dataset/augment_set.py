import numpy as np
from keras.preprocessing.image import ImageDataGenerator


#carefull, the samples will be partially ordered after this
#the new set first contains the old set, followed by the specified number of randomly permutated samples of the selected class
#num_runs_to_add = 1 means we will have twice as many samples of class class_label in the end
def augment_set(x_samples, y_samples, num_runs_to_add = 1, class_label = 1):
    
    if x_samples.shape[0] != y_samples.shape[0]:
        print('number of samples doesnt equal number of labels, something is wrong!\n')
        
    num_total_samples_before = x_samples.shape[0]
            
    x_myclass = x_samples[y_samples[:,0] == class_label,...]
    #y_myclass = y_samples[y_samples[:,0] == class_label,...]
    
    #count the number of samples to augment, to reserve the appropriate space
    num_positive_samples = x_myclass.shape[0]
    
    #reserve space for the new samples (shape is the same as before, exept with a new number of samples)
    x_augmented = np.zeros((((x_samples.shape[0] + num_positive_samples*num_runs_to_add),) + x_samples.shape[1:]), dtype='uint8')
    y_augmented = np.zeros((((y_samples.shape[0] + num_positive_samples*num_runs_to_add),) + y_samples.shape[1:]), dtype='uint8')
    
    #the first part of the set is the old set
    x_augmented[0:x_samples.shape[0],...] = x_samples
    y_augmented[0:y_samples.shape[0],...] = y_samples
    
    datagen = ImageDataGenerator(
    rotation_range=90,
    zoom_range=[1, 1.4],#[lower, upper] = [1-zoom_range, 1+zoom_range]
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='reflect',
    preprocessing_function=None) # maybe include the strech histogram here

    datagen.fit(x_myclass, augment=True, seed=42)
    
    #now iterate trough all samples of class_label num_runs_to_add times
    counter = num_total_samples_before 
    for i in  range(num_runs_to_add):
        for current_sample in x_myclass:
            #print(current_sample.shape)
            x_augmented[counter, ...] = datagen.random_transform(current_sample)
            counter += 1
 
    return x_augmented, y_augmented