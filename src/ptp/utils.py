import numpy as np
from typing import Literal
from skimage.metrics import structural_similarity as ssim


def training_data_generator(seismic: np.ndarray, axis: Literal['i_line', 'x_line', None]=None, percentile: int=25):
    """Function to delete part of original seismic volume and extract target region

    Parameters:
        seismic: np.ndarray 3D matrix with original survey
        axis: one of 'i_line','x_line' or None. Axis along which part of survey will be deleted.
              If None (default), random will be chosen
        percentile: int, size of deleted part relative to axis. Any integer between 1 and 99 (default 20)

    Returns:
        seismic: np.ndarray, original survey 3D matrix with deleted region
        target: np.ndarray, 3D deleted region
        target_mask: np.ndarray, position of target 3D matrix in seismic 3D matrix. 
                     This mask is used to reconstruct original survey -> seismic[target_mask]=target.reshape(-1)
    """

    # check parameters
    assert isinstance(seismic, np.ndarray) and len(seismic.shape)==3, 'seismic must be 3D numpy.ndarray'
    assert axis in ['i_line', 'x_line', None], 'axis must be one of: i_line, x_line or None'
    assert type(percentile) is int and 0<percentile<100, 'percentile must be an integer between 0 and 100'

    # rescale volume
    minval = np.percentile(seismic, 2)
    maxval = np.percentile(seismic, 98)
    seismic = np.clip(seismic, minval, maxval)
    seismic = ((seismic - minval) / (maxval - minval)) * 255

    # if axis is None get random choice
    if axis is None:
        axis = np.random.choice(['i_line', 'x_line'], 1)[0]

    # crop subset
    if axis == 'i_line':
        sample_size = np.round(seismic.shape[0]*(percentile/100)).astype('int')
        sample_start = np.random.choice(range(seismic.shape[0]-sample_size), 1)[0]
        sample_end = sample_start+sample_size

        target_mask = np.zeros(seismic.shape).astype('bool')
        target_mask[sample_start:sample_end, :, :] = True

        target = seismic[sample_start:sample_end, :, :].copy()
        seismic[target_mask] = np.nan

    else:
        sample_size = np.round(seismic.shape[1]*(percentile/100)).astype('int')
        sample_start = np.random.choice(range(seismic.shape[1]-sample_size), 1)[0]
        sample_end = sample_start+sample_size

        target_mask = np.zeros(seismic.shape).astype('bool')
        target_mask[:, sample_start:sample_end, :] = True

        target = seismic[:, sample_start:sample_end, :].copy()
        seismic[target_mask] = np.nan

    return seismic, target, target_mask


def scoring(prediction_path, ground_truth_path):
    """Scoring function. Use scikit-image implementation of Structural Similarity Index:
       https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity
       
    Parameters:
        prediction_path: path of perdiction .npz file
        ground_truth_path: path of ground truth .npz file

    Returns:
        score: -1 to 1 structural similarity index
    """
    
    ground_truth =  np.load(ground_truth_path)
    prediction =  np.load(prediction_path)
    
    score = np.mean([ssim(ground_truth[key], prediction[key], data_range=255) for key in ground_truth.files])
    
    return score

def create_submission(seismic_filenames: list, prediction: list, submission_path: str):
    """Function to create submission file out of all test predictions in one list

    Parameters:
        seismic_filenames: list of survey .npy filenames used for perdiction
        prediction: list with 3D np.ndarrays of predicted missing parts
        submission_path: path to save submission

    Returns:
        None
    """

    submission = dict({})
    for sample_name, sample_prediction in zip(seismic_filenames, prediction):
        i_slices_index = (np.array([.25, .5, .75]) * sample_prediction.shape[0]).astype(int)
        i_slices_names = [f'{sample_name}-i_{n}' for n in range(0,3)]
        i_slices = [sample_prediction[s, :, :].astype(np.uint8) for s in i_slices_index]
        submission.update(dict(zip(i_slices_names, i_slices)))

        x_slices_index = (np.array([.25, .5, .75]) * sample_prediction.shape[1]).astype(int)
        x_slices_names = [f'{sample_name}-x_{n}' for n in range(0,3)]
        x_slices = [sample_prediction[:, s, :].astype(np.uint8) for s in x_slices_index]
        submission.update(dict(zip(x_slices_names, x_slices)))
    
    
    np.savez(submission_path, **submission)


def create_single_submission(seismic_filename: str, prediction: np.ndarray, submission_path: str):
    """Function to create submission file out of one test prediction at time

    Parameters:
        seismic_filename: filename of survey .npy used for perdiction
        prediction: 3D np.ndarray of predicted missing part
        submission_path: path to save submission

    Returns:
        None
    """
    
    try:
        submission = dict(np.load(submission_path))
    except:
        submission = dict({})
        
    i_slices_index = (np.array([.25, .5, .75]) * prediction.shape[0]).astype(int)
    i_slices_names = [f'{seismic_filename}-i_{n}' for n in range(0,3)]
    i_slices = [prediction[s, :, :].astype(np.uint8) for s in i_slices_index]
    submission.update(dict(zip(i_slices_names, i_slices)))

    x_slices_index = (np.array([.25, .5, .75]) * prediction.shape[1]).astype(int)
    x_slices_names = [f'{seismic_filename}-x_{n}' for n in range(0,3)]
    x_slices = [prediction[:, s, :].astype(np.uint8) for s in x_slices_index]
    submission.update(dict(zip(x_slices_names, x_slices)))
    
    np.savez(submission_path, **submission)