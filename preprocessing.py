# Step 1: Perform bias field correction and skull-stripping using tools such as FSL, FreeSurfer, or SynthStrip
# ……
# ……


# Step 2:  Rigid registration to the MNI template space
# import os
# import ants
# from multiprocessing import Pool
# from tqdm import tqdm
# DATA_ROOT = "/your/path/to/data/"
# TEMPLATE_PATH = "/your/path/to/MNI152_T1_1mm_Brain.nii.gz"
# SAVE_ROOT = "/your/path/to/save/"
# # or other modality images
# T2_FILENAME = "T2-brain.nii.gz"
# NUM_WORKERS = 4
# def register_subject(args):
#     subject_dir, template_path, save_root_dir = args
#     try:
#         subject_id = os.path.basename(subject_dir)
#         t1_path = os.path.join(subject_dir, "brain.nii.gz")
#         t2_path = os.path.join(subject_dir, T2_FILENAME)
#         save_dir = os.path.join(save_root_dir, subject_id)
#         os.makedirs(save_dir, exist_ok=True)
#         save_t1_path = os.path.join(save_dir, "brain.nii.gz")
#         save_t2_path = os.path.join(save_dir, T2_FILENAME)
#         if not os.path.exists(t1_path):
#             return f"SKIP: {subject_id} (T1 not found)"
#         fixed_img = ants.image_read(template_path)
#         moving_t1 = ants.image_read(t1_path)
#         print(f"[{subject_id}] Registering T1 to Template...")
#         mytx = ants.registration(fixed=fixed_img, moving=moving_t1, type_of_transform='Rigid')
#         ants.image_write(mytx['warpedmovout'], save_t1_path)
#         if os.path.exists(t2_path):
#             moving_t2 = ants.image_read(t2_path)
#             warped_t2 = ants.apply_transforms(fixed=fixed_img, moving=moving_t2, transformlist=mytx['fwdtransforms'], interpolator='linear')
#             ants.image_write(warped_t2, save_t2_path)
#         else:
#             return f"WARNING: {subject_id} (T1 done, but T2 missing)"
#         return None
#     except Exception as e:
#         return f"ERROR: {subject_id} - {str(e)}"
# def main():
#     if not os.path.exists(DATA_ROOT):
#         return
#     subject_dirs = sorted([os.path.join(DATA_ROOT, d) for d in os.listdir(DATA_ROOT)
#                            if os.path.isdir(os.path.join(DATA_ROOT, d))])
#     if not subject_dirs:
#         return
#     tasks = []
#     for s_dir in subject_dirs:
#         tasks.append((s_dir, TEMPLATE_PATH, SAVE_ROOT))
#     error_logs = []
#     with Pool(processes=NUM_WORKERS) as pool:
#         for res in tqdm(pool.imap_unordered(register_subject, tasks), total=len(tasks), unit="subj"):
#             if res:
#                 error_logs.append(res)
# if __name__ == "__main__":
#     main()


# Step 3: Reorientation to a consistent RPI coordinate system
# import ants
# import numpy as np
# from tqdm import tqdm
# def _ants_img_info(img_path):
#     img = ants.image_read(img_path)
#     return img.origin, img.spacing, img.direction, img.numpy()
#
#
# def _LPI_2_RPI(img_path):
#     '''
#     TODO: Functions to reorient LPI to RPI
#     @param img_path: image path with LPI space
#     @return: reoriented RPI space image
#     '''
#     RPI_origin = (-90.0, 126.0, -72.0)
#     RPI_direction = (np.array
#                      ([[1., 0., 0.],
#                        [0., -1., 0.],
#                        [0., 0., 1.]]))
#     origin, spacing, direction, img = _ants_img_info(img_path)
#     img = img[::-1, :, :]
#     img = ants.from_numpy(img, RPI_origin, spacing, RPI_direction)
#     return img

# if __name__ == '__main__':
#     path = r'/your/path/to/reorient/'
#     save = r'/your/path/to/save/'
#     for name in tqdm(os.listdir(path)):
#         file_path = os.path.join(path, name)
#         for img in os.listdir(file_path):
#             img_path = os.path.join(file_path, img)
#             save_path = os.path.join(save, name)
#             new = _LPI_2_RPI(img_path)
#             os.makedirs(save_path, exist_ok=True)
#             ants.image_write(new, os.path.join(save_path, img))


# Step 4: Spatial cropping to a fixed volume size of (224, 256, 224)
import numpy as np
from scipy.ndimage import binary_fill_holes
import SimpleITK as sitk
import os
from tqdm import tqdm
import nibabel as nib
def create_nonzero_mask(data):
    """
    :param data:
    :return: the non-zero region of the image
    """
    nonzero_mask = np.zeros(data.shape, dtype=bool)
    this_mask = data != 0
    nonzero_mask = nonzero_mask | this_mask
    nonzero_mask = binary_fill_holes(nonzero_mask)
    return nonzero_mask


def get_bbox_from_mask(mask, outside_value=0):
    mask_voxel_coords = np.where(mask != outside_value)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]


def crop_to_bbox(image, bbox):
    resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
    return image[resizer]


def crop_to_nonzero(data, seg=None, nonzero_label=-1):
    """
    :param data:
    :param seg:
    :param nonzero_label: this will be written into the segmentation map
    :return:
    """
    nonzero_mask = create_nonzero_mask(data)
    bbox = get_bbox_from_mask(nonzero_mask, 0)

    cropped_data = []
    cropped = crop_to_bbox(data, bbox)
    cropped_data.append(cropped)
    data = np.vstack(cropped_data)

    if seg is not None:
        cropped_seg = []
        cropped = crop_to_bbox(seg, bbox)
        cropped_seg.append(cropped)
        seg = np.vstack(cropped_seg)
    # nonzero_mask = crop_to_bbox(nonzero_mask, bbox)
    # if seg is not None:
    #     seg[(seg == 0) & (nonzero_mask == 0)] = nonzero_label
    # else:
    #     nonzero_mask = nonzero_mask.astype(int)
    #     nonzero_mask[nonzero_mask == 0] = nonzero_label
    #     nonzero_mask[nonzero_mask > 0] = 0
    #     seg = nonzero_mask
    return data, seg, bbox


def save_cropped_image(cropped_data, original_image, save_path, patient_folder, filename):
    cropped_image = sitk.GetImageFromArray(cropped_data)
    cropped_image.SetDirection(original_image.GetDirection())
    cropped_image.SetOrigin(original_image.GetOrigin())
    cropped_image.SetSpacing(original_image.GetSpacing())
    os.makedirs(os.path.join(save_path, patient_folder), exist_ok=True)
    sitk.WriteImage(cropped_image, os.path.join(save_path, patient_folder, filename))


def batch_crop(Filepath, Savepath):
    optional_modalities = ["brain.nii.gz", "T2-brain.nii.gz", "CT-brain.nii.gz", 'DWI-brain.nii.gz', 'Flair-brain.nii.gz', 'US-brain.nii.gz', 'PD-brain.nii.gz', 'SWI-brain.nii.gz', 'T2s-brain.nii.gz']
    pet_prefixes = ["AV45", "FDG", "TAU", "Dynamic", "PIB", "CTAC", "Flumetamol", "NAV4694", "SUV", "SUM"]

    for filename in tqdm(os.listdir(Filepath)):
        if not os.path.exists(os.path.join(Savepath, filename)):
            filepath = os.path.join(Filepath, filename)
            seg_path = os.path.join(filepath, 'tissue.nii.gz')
            tissue_path = os.path.join(filepath, 'dk-struct.nii.gz')
            seg_image = sitk.ReadImage(seg_path)
            seg_data = sitk.GetArrayFromImage(seg_image)
            tissue_image = sitk.ReadImage(tissue_path)
            tissue_data = sitk.GetArrayFromImage(tissue_image)
            data, seg, bbox = crop_to_nonzero(seg_data, tissue_data)
            save_cropped_image(data, seg_image, Savepath, filename, 'tissue.nii.gz')
            save_cropped_image(seg, tissue_image, Savepath, filename, 'dk-struct.nii.gz')

            for modality in optional_modalities:
                modality_path = os.path.join(filepath, modality)
                if os.path.exists(modality_path):
                    mod_image = sitk.ReadImage(modality_path)
                    mod_data = sitk.GetArrayFromImage(mod_image)
                    cropped_modality = crop_to_bbox(mod_data, bbox)
                    save_cropped_image(cropped_modality, mod_image, Savepath, filename, modality)

            for pet_prefix in pet_prefixes:
                pet_file = f"{pet_prefix}-brain.nii.gz"
                pet_path = os.path.join(filepath, pet_file)
                if os.path.exists(pet_path):
                    pet_image = sitk.ReadImage(pet_path)
                    pet_data = sitk.GetArrayFromImage(pet_image)
                    cropped_pet = crop_to_bbox(pet_data, bbox)
                    save_cropped_image(cropped_pet, pet_image, Savepath, filename, pet_file)


def pad_image(input_file, output_folder, target_shape):
    img = nib.load(input_file)
    data = img.get_fdata()
    original_shape = data.shape

    padding = []
    for orig_dim, target_dim in zip(original_shape, target_shape):
        total_padding = max(target_dim - orig_dim, 0)
        left_padding = total_padding // 2
        right_padding = total_padding - left_padding
        padding.append((left_padding, right_padding))

    padded_data = np.pad(data, padding, mode='constant', constant_values=0)
    if padded_data.shape != target_shape:
        return

    padded_img = nib.Nifti1Image(padded_data, img.affine, img.header)
    output_file = os.path.join(output_folder, os.path.basename(input_file))
    nib.save(padded_img, output_file)


def batch_pad(input_folder, save_folder, target_shape):
    required_files = ['T2-brain.nii.gz', 'CT-brain.nii.gz', 'DWI-brain.nii.gz', 'T2-Flair.nii.gz', 'US-brain.nii.gz', 'PD-brain.nii.gz', 'SWI-brain.nii.gz', 'T2s-brain.nii.gz']
    optional_modalities = ["brain.nii.gz", "tissue.nii.gz", 'dk-struct.nii.gz']
    pet_prefixes = ["AV45", "FDG", "TAU", "Dynamic", "PIB", "CTAC", "Flumetamol", "NAV4694", "SUV", "SUM"]

    for folder_name in tqdm(os.listdir(input_folder)):
        folder_path = os.path.join(input_folder, folder_name)
        save_path = os.path.join(save_folder, folder_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

            for filename in required_files:
                file_path = os.path.join(folder_path, filename)
                pad_image(file_path, save_path, target_shape)

            for modality in optional_modalities:
                modality_path = os.path.join(folder_path, modality)
                if os.path.exists(modality_path):
                    pad_image(modality_path, save_path, target_shape)

            for pet_prefix in pet_prefixes:
                pet_file = f"{pet_prefix}-brain.nii.gz"
                pet_path = os.path.join(folder_path, pet_file)
                if os.path.exists(pet_path):
                    pad_image(pet_path, save_path, target_shape)

if __name__ == '__main__':
    batch_crop(r'/your/path/to/crop/', r'/your/path/to/save/')
    batch_pad(r'/your/path/to/pad/', r'/your/path/to/save/', (224, 256, 224))