import torch
import numpy as np

import cv2
import pydicom

import time

from planar import BoundingBox
from PIL import Image, ImageFile
from skimage import transform

ImageFile.LOAD_TRUNCATED_IMAGES = True


def normalize_image_data(tensor_frames, normalization_values):

    average = normalization_values[0]
    std = normalization_values[1]

    normalized_frames = []

    for frame in tensor_frames:
        frame = frame.cpu().numpy()
        binnary_mask = frame[0]
        frame_copy_one =  (frame[1] - float(average)) / float(std)
        frame_copy_two =  (frame[2] - float(average)) / float(std)
        merged_frame_data = [binnary_mask, frame_copy_one,frame_copy_two]
        normalized_frames.append(merged_frame_data)

    return torch.tensor(np.array(normalized_frames))



def get_preprocessed_frames(path, fps, pulse, orientation):
    start_time = time.time()
    print('Preprocessing ' + path + '...', end='')

    min_frame_number = 40
    min_heart_rate = 30
    max_heart_rate = 150
    num_of_images = 20

    avg = 25.44
    std = 44.87

    # Load data from DICOM file
    dataset = pydicom.dcmread(path, force=True)

    # Convert frames to grayscale frames
    gray_frames = np.zeros(dataset.pixel_array.shape)
    gray_frames = dataset.pixel_array[:, :, :, 0]

    # Flip videos
    if orientation == 'Stanford':
        for i, frame in enumerate(gray_frames):
            gray_frames[i] = cv2.flip(frame, 1)

    shape_of_frames = gray_frames.shape
    changes = np.zeros((shape_of_frames[1], shape_of_frames[2]))
    changes_frequency = np.zeros((shape_of_frames[1], shape_of_frames[2]))
    binary_mask = np.zeros((shape_of_frames[1], shape_of_frames[2]))
    cropped_frames = []

    for i in range(len(gray_frames) - 1):
        diff = abs(gray_frames[i] - gray_frames[i + 1])
        changes += diff
        nonzero = np.nonzero(diff)
        changes_frequency[nonzero[0], nonzero[1]] += 1

    max_of_changes = np.amax(changes)
    min_of_changes = np.min(changes)

    for r in range(len(changes)):
        for p in range(len(changes[r])):
            if int(changes_frequency[r][p]) < 10:
                changes[r][p] = 0
            else:
                changes[r][p] = int(255 * ((changes[r][p] - min_of_changes) / (max_of_changes - min_of_changes)))

    nonzero_values_for_binary_mask = np.nonzero(changes)

    binary_mask[nonzero_values_for_binary_mask[0], nonzero_values_for_binary_mask[1]] += 1
    kernel = np.ones((5, 5), np.int32)
    erosion_on_binary_msk = cv2.erode(binary_mask, kernel, iterations=1)
    binary_mask_after_erosion = np.where(erosion_on_binary_msk, binary_mask, 0)

    nonzero_values_after_erosion = np.nonzero(binary_mask_after_erosion)
    binary_mask_coordinates = np.array([nonzero_values_after_erosion[0], nonzero_values_after_erosion[1]]).T
    binary_mask_coordinates = list(map(tuple, binary_mask_coordinates))
    bbox = BoundingBox(binary_mask_coordinates)
    cropped_mask = binary_mask_after_erosion[int(bbox.min_point.x):int(bbox.max_point.x),
                   int(bbox.min_point.y):int(bbox.max_point.y)]

    for row in cropped_mask:
        ids = [i for i, x in enumerate(row) if x == 1]
        if len(ids) < 2:
            continue
        row[ids[0]:ids[-1]] = 1

    for i in range(len(gray_frames)):
        masked_image = np.where(erosion_on_binary_msk, gray_frames[i], 0)
        cropped_image = masked_image[int(bbox.min_point.x):int(bbox.max_point.x),
                        int(bbox.min_point.y):int(bbox.max_point.y)]
        cropped_frames.append(cropped_image)

    if pulse is not None:
        if pulse < min_heart_rate or pulse > max_heart_rate:
            raise ValueError('Heart rate is out of boundary! It should be in the {} - {} range'.format(min_heart_rate,
                                                                                                       max_heart_rate))
        else:
            heart_rate = pulse
    else:
        if hasattr(dataset, 'HeartRate') and (min_heart_rate < dataset.HeartRate < max_heart_rate):
            heart_rate = dataset.HeartRate
        else:
            raise ValueError(
                'Heart rate is out of boundary! It should be in the {} - {} range'.format(min_heart_rate,
                                                                                          max_heart_rate))

    if fps is None:
        if hasattr(dataset, 'RecommendedDisplayFrameRate'):
            fps = dataset.RecommendedDisplayFrameRate
        else:
            raise ValueError(
                'FPS not found in DICOM, please provide an FPS value!')

    print('FPS: {}, pulse: {}'.format(fps, heart_rate))
    len_of_dicom = len(dataset.pixel_array)
    if len_of_dicom < min_frame_number:
        raise ValueError('Number of frames in the recording is less than 40!')
    len_of_heart_cycle = 60 / int(heart_rate) * int(float(fps))
    sampling_frequency = len_of_heart_cycle / num_of_images
    nbr_valid_cycles = int(len(cropped_frames)/len_of_heart_cycle)

    # Sample frames from multiple heart cycle:

    hear_cycle_data = []
    start_index = 0

    for cycle_idx in range(nbr_valid_cycles):

        sampled_indexes = np.arange(start_index, (cycle_idx+1)*int(len_of_heart_cycle), sampling_frequency)
        start_index = sampled_indexes[-1] + sampling_frequency
        sampled_indexes = list([int(i) for i in sampled_indexes])

        
        
        selected_frames = [cropped_frames[i] for i in sampled_indexes]

        # Resize frames and binary mask
        resized_frames = []
        for frame in selected_frames:
            resized_frame = transform.resize(frame, (224, 224))
            resized_frames.append(resized_frame)
        resized_frames = np.asarray(resized_frames)
        resized_binary_mask = transform.resize(cropped_mask, (224, 224))

        # Convert 1 channel image to 3 channel image
        frames_3ch = []
        for frame in resized_frames:
            new_frame = np.zeros((np.array(frame).shape[0], np.array(frame).shape[1], 3))
            new_frame[:, :, 0] = frame
            new_frame[:, :, 1] = frame
            new_frame[:, :, 2] = frame
            frames_3ch.append(new_frame)

        # ToTensor
        frames_tensor = np.array(frames_3ch)
        #print(frames_tensor.shape)
        frames_tensor = frames_tensor.transpose((0, 3, 1, 2))
        binary_mask_tensor = np.array(resized_binary_mask)
        frames_tensor = torch.from_numpy(frames_tensor)
        binary_mask_tensor = torch.from_numpy(binary_mask_tensor)

        # Expand Frame Tensor
        f, c, h, w = frames_tensor.size()
        new_shape = (f, 3, h, w)

        expanded_frames = frames_tensor.expand(new_shape)
        expanded_frames_clone = expanded_frames.clone()
        expanded_frames_clone[:, 0, :, :] = binary_mask_tensor

        hear_cycle_data.append(expanded_frames_clone)

    hear_cycle_data_tensor = torch.stack(hear_cycle_data)

    print('done!')
    print('--- %s seconds ---' % (time.time() - start_time))

    return hear_cycle_data_tensor
