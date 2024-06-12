import os
import random
import cv2
import numpy as np
import skvideo.io as skvio
import torch

from src.utility.file import check_file_path


def get_video_properties(vpath: str):
    """ Returns a dictionary with following video properties,
    {
        'islocal': boolean,
        'full_path': str,
        'name': str,
        'extension': str,
        'dir_loc': str,
        'frame_rate': int,
        'duration': int,
        'num_frames': int,
        'width': int,
        'height': int,
        'frame_dim': tuple 
    }

    Parameters
    ----------
    vpath: str
        Video file path
    """
    # Get video file name and directory location
    vdir_loc = os.path.dirname(vpath)
    vname, vext = os.path.splitext(os.path.basename(vpath))

    # Read video meta information
    vmeta = skvio.ffprobe(vpath)

    # If it is empty i.e. scikit video cannot read metadata
    # return empty stings and zeros
    if vmeta == {}:
        vprops = {
            'islocal': False,
            'full_path': vpath,
            'name': vname,
            'extension': vext,
            'dir_loc': vdir_loc,
            'frame_rate': 0,
            'duration': 0,
            'num_frames': 0,
            'width': 0,
            'height': 0,
            'frame_dim': None
        }

        return vprops

    # Calculate average frame rate
    fr_str = vmeta['video']['@avg_frame_rate']
    fr = round(int(fr_str.split("/")[0]) / int(fr_str.split("/")[1]))

    # get duration
    vdur = round(float(vmeta['video']['@duration']))

    # get number of frames
    vnbfrms = int(vmeta['video']['@nb_frames'])

    # video width
    width = int(vmeta['video']['@width'])

    # video height
    height = int(vmeta['video']['@height'])

    # Frame dimension assuming color video
    frame_dim = (height, width, 3)

    # Creating properties dictionary
    vprops = {
        'islocal': True,
        'full_path': vpath,
        'name': vname,
        'extension': vext,
        'dir_loc': vdir_loc,
        'frame_rate': fr,
        'duration': vdur,
        'num_frames': vnbfrms,
        'width': width,
        'height': height,
        'frame_dim': frame_dim
    }

    return vprops

def save_spatiotemporal_trim(video_props, sfrm, efrm, bbox, opth):
        """
        Create a spatiotemporl trim. The output video name is
        <in_vid>_sfrm_efrm.mp4

        Parameters
        ----------
        sfrm: int
            Frame number of starting frame.
        efrm: int
            Frame number of ending frame.
        bbox: int[arr]
            Bounding box,
            [<width_location>, <height_location>, <width>, <height>]
        opth: str
            Output video path
        """
        
        # Time stamps from frame numbers
        sts = sfrm / video_props['frame_rate']
        nframes = efrm - sfrm

        # Creating ffmpeg command string
        crop_str = f"{bbox[2]}:{bbox[3]}:{bbox[0]}:{bbox[1]}"
        ffmpeg_cmd = (
            f'ffmpeg -hide_banner -loglevel warning '
            f'-y -ss {sts} -i {video_props["full_path"]} -vf "crop={crop_str}" '
            f'-c:v libx264 -crf 0 -frames:v {nframes} {opth}')
        os.system(ffmpeg_cmd)

        return opth

def load_to_tensor_using_cv2(vpath, oshape, data_aug_flag = False):
    """ Loads a video as tensor using OpenCV

    Parameters
    ----------
    oshape: tuple of ints
        (output width, output height)
    data_aug_flag : bool
        Data augmentation flag
    """
    props = {}
    if not check_file_path(vpath):
        props['islocal'] = False

    # Get video properties
    props = get_video_properties(vpath)

    # Initialize torch tensor that can contain video
    frames_torch = torch.FloatTensor(
        3, props['num_frames'], oshape[1], oshape[0]
    )

    # Initialize OpenCV video object
    vo = cv2.VideoCapture(props['full_path'])

    if data_aug_flag:
        # Augmentation probability per video. The values are derived
        # from Sravani's thesis
        # 1. Rotation, {-7,...,+7}
        # 2. w_translation = {-20...+20} for Width = 858
        #                  = {-5,...+5} for Width = 224
        # 3. Flip with a probability of 0.5
        # 4. Rescaling the frame between [0.8 to 1.2]
        # 5. Shearing with x axis witht a factor of [-0.05, 0.05] <--- I eyed this not from sravani thesis
        aug_prob = random.uniform(0,1)
        if aug_prob > 0.5:
            shear_factor = random.uniform(-0.05, 0.05)
            rescaling_ratio = round(random.uniform(0.8, 1.2), 1)
            rot_angle = random.randint(-7, 7)
            w_translation = random.randint(-5, 5)
            hflip_prob = random.uniform(0,1)

    poc = 0  # picture order count
    while vo.isOpened():
        ret, frame = vo.read()
        if ret:
            frame = cv2.resize(frame, oshape)
            frame_orig = frame.copy()
            if data_aug_flag:
                if aug_prob > 0.5:
                    frame = apply_horizontal_flip(frame, hflip_prob)
                    frame = apply_scaling(frame, rescaling_ratio)
                    frame = apply_shearing(frame, shear_factor)
                    frame = apply_rotation(frame, rot_angle)
                    frame = apply_horizontal_translation(frame, w_translation)
                    # cv2.imshow("orig", frame_orig)
                    # cv2.imshow("aug", frame)
                    # cv2.waitKey(0)


            frame_torch = torch.from_numpy(frame)
            frame_torch = frame_torch.permute(
                2, 0, 1)  # (ht, wd, ch) to (ch, ht, wd)
            frames_torch[:, poc, :, :] = frame_torch
            poc += 1
        else:
            break

    vo.release()
    return frames_torch


def apply_horizontal_flip(frame, hflip_prob):
        """Applies horizontal flip with a certain probability

        Parameters
        ----------
        frame : ndarray
            Input RGB frame
        flip_prob : float
            Flip probability.
            
        """
        if hflip_prob > 0:
            frame_out = cv2.flip(frame, 1)

        # cv2.imshow("no flip", frame)
        # cv2.imshow("flip", frame_out)
        # cv2.waitKey(0)
        # sys.exit()
        return frame_out


def apply_shearing(frame, shear_factor):
        """Applies horizontal flip with a certain probability

        Parameters
        ----------
        frame : ndarray
            Input RGB frame
        shear_factor : float
            shearing factor
            
        """
        rows, cols, dim = frame.shape
        M = np.float32(
            [[1, shear_factor, 0],
             [0, 1  , 0],
             [0, 0  , 1]]
        )
        frame_out = cv2.warpPerspective(frame,M,(rows,cols))

        # cv2.imshow("no shear", frame)
        # cv2.imshow(f"x shear {shear_factor}", frame_out)
        # cv2.waitKey(0)
        # sys.exit()
        return frame_out

        
        

def apply_scaling(frame, scaling_ratio):
        """Applies horizontal flip with a certain probability

        Parameters
        ----------
        frame : ndarray
            Input RGB frame
        scaling_ratio : float
            Scaling ratio.
            
        """
        h, w = frame.shape[:2]
        frame_out = cv2.resize(
            frame, None, fx=scaling_ratio, fy=scaling_ratio, interpolation = cv2.INTER_CUBIC
        )
        h_, w_ = frame_out.shape[:2]
        
        if scaling_ratio >= 1:
            frame_out = frame_out[
                int(h_/2 - h/2): int(h_/2 + h/2),
                int(w_/2 - w/2): int(w_/2 + w/2)
            ]
        else:
            zero_img = np.zeros((h, w, 3), dtype=np.uint8)
            zero_img[ int(h/2 - h_/2) : int(h/2 + h_/2), int(w/2 - w_/2)  : int(w/2 + w_/2)] = frame_out
            frame_out = zero_img

        # cv2.imshow("no flip", frame)
        # cv2.imshow("flip", frame_out)
        # cv2.waitKey(0)
        # sys.exit()
            
        return frame_out
    

def apply_rotation(frame, rot_angle):
        """Applies rotation with optimal values from Sravani's
        thesis. Rotation from -8 to +8 with a probability of 0.5.

        Parameters
        ----------
        frame : ndarray
            Input RGB frame
        rot_angle : int
            Rotation angle in degrees
        """
        (h, w) = frame.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), rot_angle, 1.0)
        frame_out = cv2.warpAffine(frame, M, (w, h))
        return frame_out

def apply_horizontal_translation(frame, w_translation):
        """Applies translation in width direction (horizontal).

        Parameters
        ----------
        frame : ndarray
            Input RGB frame
        w_translation : int
            Translation to be done in x axis
        """
        # get the width and height of the image
        height, width = frame.shape[:2]
        tx, ty = w_translation, 0
        # create the translation matrix using tx and ty, it is a NumPy array 
        translation_matrix = np.array([
            [1, 0, tx],
            [0, 1, ty]
        ], dtype=np.float32)
        frame_out = cv2.warpAffine(
            src=frame,
            M=translation_matrix,
            dsize=(width, height)
        )
        # cv2.imshow("no translate", frame)
        # cv2.imshow(f"translation {w_translation}", frame_out)
        # cv2.waitKey(0)
        # sys.exit()
        return frame_out