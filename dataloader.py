import os
import pandas as pd
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn import functional as F
from BCLIP.textencoder import load_text_encoder
from monai.transforms import Compose, NormalizeIntensityd, RandSpatialCropd
import re
import random
from itertools import chain, combinations


def process_image(image):
    image = image[None, None, :, :, :]
    tensor_image = torch.tensor(image).float()
    image_dict = {"image": tensor_image}
    transforms = Compose([NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True)])
    image_dict = transforms(image_dict)
    image = image_dict['image'].squeeze(0)
    return image


def process_label(label):
    label = label[None, None, :, :, :]
    tensor_label = torch.tensor(label).float()
    label = tensor_label.squeeze(0)
    return label


def get_pet(modality):
    modality = modality.strip()
    modality = modality.split('[')[1].strip(' ]').split(', ')
    pet_type = None
    for m in modality:
        if 'PET' in m:
            pet_type = m.split(' PET')[0]
            break
    return pet_type


# calculate sobel edge map
def sobel_3d(image):
    sobel_x = torch.tensor([[[[-1, 0, 1], [-3, 0, 3], [-1, 0, 1]],
                             [[-3, 0, 3], [-6, 0, 6], [-3, 0, 3]],
                             [[-1, 0, 1], [-3, 0, 3], [-1, 0, 1]]]], dtype=torch.float32)
    sobel_y = torch.tensor([[[[-1, -3, -1], [0, 0, 0], [1, 3, 1]],
                             [[-3, -6, -3], [0, 0, 0], [3, 6, 3]],
                             [[-1, -3, -1], [0, 0, 0], [1, 3, 1]]]], dtype=torch.float32)

    sobel_z = torch.tensor([[[[-1, -3, -1], [-3, -6, -3], [-1, -3, -1]],
                             [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                             [[1, 3, 1], [3, 6, 3], [1, 3, 1]]]], dtype=torch.float32)
    image = image[None]
    grad_x = F.conv3d(image, sobel_x[None], padding=1)
    grad_y = F.conv3d(image, sobel_y[None], padding=1)
    grad_z = F.conv3d(image, sobel_z[None], padding=1)
    edge_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + grad_z ** 2)
    edge = edge_magnitude.squeeze(0)
    return edge


class SegDataset(Dataset):
    def __init__(self, texts_path, images_path, mode_prob=0.5):
        self.data = pd.read_excel(texts_path)
        self.images_path = images_path
        self.tokenizer, _ = load_text_encoder()
        self.mode_prob = mode_prob

    def __len__(self):
        return len(self.data)

    def update_caption(self, caption, modality_list):
        caption_parts = caption.split(';')
        modalities = caption_parts[2].strip()
        modalities = modalities[modalities.index('[') + 1:modalities.index(']')].split(', ')
        # If a single mode is activated (that is, there is only one non-None mode)
        # Then only the position corresponding to this mode is retained and remains unchanged, while the other positions are replaced with None
        if modality_list.count(None) == 4:
            for i in range(len(modalities)):
                if modality_list[i] is None:
                    modalities[i] = "None"
            modalities = "[" + ", ".join(modalities) + "]"
            caption_parts[2] = " Modality: " + modalities
            caption = ";".join(caption_parts)
        return caption

    def __getitem__(self, idx):
        caption = self.data.iloc[idx]['caption']
        patient_id = self.data.iloc[idx]['patient_id']
        affine = nib.load(os.path.join(self.images_path, patient_id, "dk-struct.nii.gz")).affine
        # tissue和dk label
        dk_struct = nib.load(os.path.join(self.images_path, patient_id, "dk-struct.nii.gz")).get_fdata()
        dk_struct = process_label(dk_struct)
        tissue = nib.load(os.path.join(self.images_path, patient_id, "tissue.nii.gz")).get_fdata()
        tissue = process_label(tissue)
        modalities = {
            "T1w MRI": None,
            "T2w MRI": None,
            "three": None,
            "PET": None,
            "Random": None,
        }
        # load T1, T2 modality
        if os.path.exists(os.path.join(self.images_path, patient_id, "brain.nii.gz")):
            modalities["T1w MRI"] = nib.load(os.path.join(self.images_path, patient_id, "brain.nii.gz")).get_fdata()
            modalities["T1w MRI"] = process_image(modalities["T1w MRI"])
        if os.path.exists(os.path.join(self.images_path, patient_id, "T2-brain.nii.gz")):
            modalities["T2w MRI"] = nib.load(os.path.join(self.images_path, patient_id, "T2-brain.nii.gz")).get_fdata()
            modalities["T2w MRI"] = process_image(modalities["T2w MRI"])
        # processing the third channel, including CT, FLAIR, and DWI
        if os.path.exists(os.path.join(self.images_path, patient_id, "CT-brain.nii.gz")):
            modalities["three"] = nib.load(os.path.join(self.images_path, patient_id, "CT-brain.nii.gz")).get_fdata()
            modalities["three"] = process_image(modalities["three"])
        elif os.path.exists(os.path.join(self.images_path, patient_id, "Flair-brain.nii.gz")):
            modalities["three"] = nib.load(os.path.join(self.images_path, patient_id, "Flair-brain.nii.gz")).get_fdata()
            modalities["three"] = process_image(modalities["three"])
        elif os.path.exists(os.path.join(self.images_path, patient_id, "DWI-brain.nii.gz")):
            modalities["three"] = nib.load(os.path.join(self.images_path, patient_id, "DWI-brain.nii.gz")).get_fdata()
            modalities["three"] = process_image(modalities["three"])
        # load PET modality
        pet_type = get_pet(caption.split(';')[2])
        if pet_type is not None:
            modalities["PET"] = nib.load(os.path.join(self.images_path, patient_id, pet_type + "-brain.nii.gz")).get_fdata()
            modalities["PET"] = process_image(modalities["PET"])
        # load ultrasound、PD、SWI or T2star modality
        if os.path.exists(os.path.join(self.images_path, patient_id, "US-brain.nii.gz")):
            modalities["Random"] = nib.load(os.path.join(self.images_path, patient_id, "US-brain.nii.gz")).get_fdata()
            modalities["Random"] = process_image(modalities["Random"])
        elif os.path.exists(os.path.join(self.images_path, patient_id, "PD-brain.nii.gz")):
            modalities["Random"] = nib.load(os.path.join(self.images_path, patient_id, "PD-brain.nii.gz")).get_fdata()
            modalities["Random"] = process_image(modalities["Random"])
        elif os.path.exists(os.path.join(self.images_path, patient_id, "SWI-brain.nii.gz")):
            modalities["Random"] = nib.load(os.path.join(self.images_path, patient_id, "SWI-brain.nii.gz")).get_fdata()
            modalities["Random"] = process_image(modalities["Random"])
        elif os.path.exists(os.path.join(self.images_path, patient_id, "T2s-brain.nii.gz")):
            modalities["Random"] = nib.load(os.path.join(self.images_path, patient_id, "T2s-brain.nii.gz")).get_fdata()
            modalities["Random"] = process_image(modalities["Random"])
        # Randomly decide whether to load all available modality or one modality
        if random.random() < self.mode_prob:
            valid_modalities = {key: value for key, value in modalities.items() if value is not None}
            selected_key = random.choice(list(valid_modalities.keys()))
            selected_modalities = {key: (modalities[key] if key == selected_key else None) for key in modalities}

        else:
            selected_modalities = {key: modalities[key] for key in modalities}

        # edge map
        edge = None
        if selected_modalities["T1w MRI"] is not None:
            edge = sobel_3d(selected_modalities["T1w MRI"])
        elif selected_modalities["T2w MRI"] is not None:
            edge = sobel_3d(selected_modalities["T2w MRI"])
        elif selected_modalities["three"] is not None:
            edge = sobel_3d(selected_modalities["three"])
        elif selected_modalities["PET"] is not None:
            edge = sobel_3d(selected_modalities["PET"])
        elif selected_modalities["Random"] is not None:
            edge = sobel_3d(selected_modalities["Random"])
        edge = process_image(edge.squeeze(0).numpy())

        channels = []
        modality_list = []
        for key in ["T1w MRI", "T2w MRI", "three", "PET", "Random"]:
            image = selected_modalities[key]
            if image is not None:
                channels.append(image)
                modality_list.append(key)
            else:
                noise = torch.randn(tissue.shape)
                channels.append(noise)
                modality_list.append(None)
        channels.append(edge)

        # whether to randomly crop is determined by your GPU memory
        # patch_size = (192, 192, 192)
        # transform = Compose([
        #     RandSpatialCropd(keys=["T1", "T2", "three", "PET", "Random", "edge", "tissue", "dk_struct"],
        #                      roi_size=patch_size, random_size=False)
        # ])

        # image_dict = {
        #     "T1": channels[0],
        #     "T2": channels[1],
        #     "three": channels[2],
        #     "PET": channels[3],
        #     "Random": channels[4],
        #     "edge": channels[5],
        #     "tissue": tissue,
        #     "dk_struct": dk_struct
        # }

        # image_dict = transform(image_dict)
        images = torch.cat([image_dict[key] for key in ["T1", "T2", "three", "PET", "Random", "edge"]], dim=0)

        # If a single modality image is randomly selected, update the the caption
        updated_caption = self.update_caption(caption, modality_list)
        text = self.tokenizer(updated_caption)[0]

        item = {
            'images': images,
            'tissue': image_dict['tissue'],
            'dk-struct': image_dict['dk_struct'],
            'text': text,
            'affine': affine,
            'ID': patient_id,
            'caption': caption,
            'updated_caption': updated_caption
        }

        return item


# Used for multimodal tissue inference
class MultiDataset(Dataset):
    def __init__(self, texts_path, images_path, index='Sheet1'):
        self.data = pd.read_excel(texts_path, sheet_name=index)
        self.images_path = images_path
        self.tokenizer, _ = load_text_encoder()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        caption = self.data.iloc[idx]['caption']
        patient_id = self.data.iloc[idx]['patient_id']
        patient_id = str(patient_id)
        affine = nib.load(os.path.join(self.images_path, patient_id, "dk-struct.nii.gz")).affine
        dk_struct = nib.load(os.path.join(self.images_path, patient_id, "dk-struct.nii.gz")).get_fdata()
        dk_struct = process_label(dk_struct)
        tissue = nib.load(os.path.join(self.images_path, patient_id, "tissue.nii.gz")).get_fdata()
        tissue = process_label(tissue)
        T1 = None
        if os.path.exists(os.path.join(self.images_path, patient_id, "brain.nii.gz")):
            T1 = nib.load(os.path.join(self.images_path, patient_id, "brain.nii.gz")).get_fdata()
            T1 = process_image(T1)
        T2 = None
        if os.path.exists(os.path.join(self.images_path, patient_id, "T2-brain.nii.gz")):
            T2 = nib.load(os.path.join(self.images_path, patient_id, "T2-brain.nii.gz")).get_fdata()
            T2 = process_image(T2)
        three = None
        if os.path.exists(os.path.join(self.images_path, patient_id, "CT-brain.nii.gz")):
            three = nib.load(os.path.join(self.images_path, patient_id, "CT-brain.nii.gz")).get_fdata()
            three = process_image(three)
        elif os.path.exists(os.path.join(self.images_path, patient_id, "Flair-brain.nii.gz")):
            three = nib.load(os.path.join(self.images_path, patient_id, "Flair-brain.nii.gz")).get_fdata()
            three = process_image(three)
        elif os.path.exists(os.path.join(self.images_path, patient_id, "DWI-brain.nii.gz")):
            three = nib.load(os.path.join(self.images_path, patient_id, "DWI-brain.nii.gz")).get_fdata()
            three = process_image(three)
        PET = None
        pet_type = get_pet(caption.split(';')[2])
        if pet_type is not None:
            PET = nib.load(os.path.join(self.images_path, patient_id, pet_type + "-brain.nii.gz")).get_fdata()
            PET = process_image(PET)
        Random = None
        if os.path.exists(os.path.join(self.images_path, patient_id, "US-brain.nii.gz")):
            Random = nib.load(os.path.join(self.images_path, patient_id, "US-brain.nii.gz")).get_fdata()
            Random = process_image(Random)
        elif os.path.exists(os.path.join(self.images_path, patient_id, "PD-brain.nii.gz")):
            Random = nib.load(os.path.join(self.images_path, patient_id, "PD-brain.nii.gz")).get_fdata()
            Random = process_image(Random)
        elif os.path.exists(os.path.join(self.images_path, patient_id, "SWI-brain.nii.gz")):
            Random = nib.load(os.path.join(self.images_path, patient_id, "SWI-brain.nii.gz")).get_fdata()
            Random = process_image(Random)
        elif os.path.exists(os.path.join(self.images_path, patient_id, "T2s-brain.nii.gz")):
            Random = nib.load(os.path.join(self.images_path, patient_id, "T2s-brain.nii.gz")).get_fdata()
            Random = process_image(Random)
        # calculate edge map
        if T1 is not None:
            edge = sobel_3d(T1)
        elif T2 is not None:
            edge = sobel_3d(T2)
        elif three is not None:
            edge = sobel_3d(three)
        elif PET is not None:
            edge = sobel_3d(PET)
        elif Random is not None:
            edge = sobel_3d(Random)
        edge = process_image(edge.squeeze(0).numpy())
        # concat channels
        channels = []
        if T1 is not None:
            channels.append(T1)
        else:
            noise = torch.randn(tissue.shape)
            channels.append(noise)
        if T2 is not None:
            channels.append(T2)
        else:
            noise = torch.randn(tissue.shape)
            channels.append(noise)
        if three is not None:
            channels.append(three)
        else:
            noise = torch.randn(tissue.shape)
            channels.append(noise)
        if PET is not None:
            channels.append(PET)
        else:
            noise = torch.randn(tissue.shape)
            channels.append(noise)
        if Random is not None:
            channels.append(Random)
        else:
            noise = torch.randn(tissue.shape)
            channels.append(noise)
        channels.append(edge)

        image_dict = {
            "T1": channels[0],
            "T2": channels[1],
            "three": channels[2],
            "PET": channels[3],
            "Random": channels[4],
            "edge": channels[5],
            "tissue": tissue,
            "dk_struct": dk_struct
        }

        # Combine images into a tensor
        images = torch.cat([image_dict[key] for key in ["T1", "T2", "three", "PET", "Random", "edge"]], dim=0)

        text = self.tokenizer(caption)[0]
        item = {
            'images': images,
            'tissue': image_dict['tissue'],
            'dk-struct': image_dict['dk_struct'],
            'text': text,
            'affine': affine,
            'ID': patient_id,
        }

        return item


# Used for unimodal tissue inference
class SingleDataset(Dataset):
    def __init__(self, texts_path, images_path, modality):
        self.data = pd.read_excel(texts_path)
        self.data.set_index('patient_id', inplace=True)
        self.images_path = images_path
        self.modality = modality
        if modality == 'PET':
            pet_prefixes = ["AV45", "FDG", "TAU", "Dynamic", "PIB", "CTAC", "Flumetamol", "NAV4694", "SUV", "SUM"]
            self.pet_prefixes = pet_prefixes
            self.images_list = [folder for folder in os.listdir(images_path) if any(os.path.exists(os.path.join(images_path, folder, f"{prefix}-brain.nii.gz")) for prefix in pet_prefixes) and folder in self.data.index]
        else:
            self.images_list = [folder for folder in os.listdir(images_path) if os.path.exists(os.path.join(images_path, folder, modality)) and folder in self.data.index]
        self.tokenizer, _ = load_text_encoder()

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        patient_id = self.images_list[idx]
        caption = self.data.loc[patient_id, 'caption']
        patient_path = os.path.join(self.images_path, patient_id)
        affine = nib.load(os.path.join(patient_path, "dk-struct.nii.gz")).affine
        dk_struct = nib.load(os.path.join(patient_path, "dk-struct.nii.gz")).get_fdata()
        dk_struct = process_label(dk_struct)
        tissue = nib.load(os.path.join(patient_path, "tissue.nii.gz")).get_fdata()
        tissue = process_label(tissue)
        if self.modality == 'PET':
            Modality = nib.load(next(os.path.join(patient_path, f"{prefix}-brain.nii.gz") for prefix in self.pet_prefixes if os.path.exists(os.path.join(patient_path, f"{prefix}-brain.nii.gz")))).get_fdata()
        else:
            Modality = nib.load(os.path.join(patient_path, self.modality)).get_fdata()
        Modality = process_image(Modality)
        edge = sobel_3d(Modality)
        edge = process_image(edge.squeeze(0).numpy())

        noise1 = torch.randn(dk_struct.shape)
        noise2 = torch.randn(dk_struct.shape)
        noise3 = torch.randn(dk_struct.shape)
        noise4 = torch.randn(dk_struct.shape)

        channels = []
        if self.modality == 'brain.nii.gz':
            channels.append(Modality)
            channels.append(noise1)
            channels.append(noise2)
            channels.append(noise3)
            channels.append(noise4)
        elif self.modality == 'T2-brain.nii.gz':
            channels.append(noise1)
            channels.append(Modality)
            channels.append(noise2)
            channels.append(noise3)
            channels.append(noise4)
        elif self.modality in ['CT-brain.nii.gz', 'DWI-brain.nii.gz', 'Flair-brain.nii.gz']:
            channels.append(noise1)
            channels.append(noise2)
            channels.append(Modality)
            channels.append(noise3)
            channels.append(noise4)
        elif self.modality == 'PET':
            channels.append(noise1)
            channels.append(noise2)
            channels.append(noise3)
            channels.append(Modality)
            channels.append(noise4)
        elif self.modality in ['US-brain.nii.gz', 'PD-brain.nii.gz', 'SWI-brain.nii.gz', 'T2s-brain.nii.gz']:
            channels.append(noise1)
            channels.append(noise2)
            channels.append(noise3)
            channels.append(noise4)
            channels.append(Modality)
        channels.append(edge)

        image_dict = {
            "T1": channels[0],
            "T2": channels[1],
            "three": channels[2],
            "PET": channels[3],
            "Random": channels[4],
            "edge": channels[5],
            "tissue": tissue,
            "dk_struct": dk_struct
        }

        images = torch.cat([image_dict[key] for key in ["T1", "T2", "three", "PET", "Random", "edge"]], dim=0)

        pattern = r'Modality: \[(.*?)\]'
        match = re.search(pattern, caption)
        modality_list = match.group(1).split(', ')
        if self.modality == 'brain.nii.gz':
            modality_list = ['T1w MRI' if i == 0 else 'None' for i in range(len(modality_list))]
        elif self.modality == 'T2-brain.nii.gz':
            modality_list = ['T2w MRI' if i == 1 else 'None' for i in range(len(modality_list))]
        elif self.modality in ['CT-brain.nii.gz', 'DWI-brain.nii.gz', 'Flair-brain.nii.gz']:
            modality_list = [f'{self.modality.split("-")[0]}' if i == 2 else 'None' for i in range(len(modality_list))]
        elif self.modality == 'PET':
            prefix = os.path.basename(next(os.path.join(patient_path, f"{prefix}-brain.nii.gz") for prefix in self.pet_prefixes if os.path.exists(os.path.join(patient_path, f"{prefix}-brain.nii.gz")))).split('-', 1)[0]
            modality_list = [f'{prefix.split("-")[0]} PET' if i == 3 else 'None' for i in range(len(modality_list))]
        elif self.modality in ['US-brain.nii.gz', 'PD-brain.nii.gz', 'SWI-brain.nii.gz', 'T2s-brain.nii.gz']:
            modality_list = ['Random' if i == 4 else 'None' for i in range(len(modality_list))]
        new_modality = f"Modality: [{', '.join(modality_list)}]"
        caption = re.sub(pattern, new_modality, caption)
        text = self.tokenizer(caption)[0]

        item = {
            'images': images,
            'tissue': image_dict['tissue'],
            'dk-struct': image_dict['dk_struct'],
            'text': text,
            'affine': affine,
            'ID': patient_id
        }

        return item


# Used for mutilmodal dk inference
class DKDataset(Dataset):
    def __init__(self, texts_path, images_path, index='Sheet1'):
        self.data = pd.read_excel(texts_path, sheet_name=index)
        self.images_path = images_path
        self.tokenizer, _ = load_text_encoder()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        caption = self.data.iloc[idx]['caption']
        patient_id = self.data.iloc[idx]['patient_id']
        patient_id = str(patient_id)
        affine = nib.load(os.path.join(self.images_path, patient_id, "dk-struct.nii.gz")).affine
        dk_struct = nib.load(os.path.join(self.images_path, patient_id, "dk-struct.nii.gz")).get_fdata()
        dk_struct = process_label(dk_struct)
        # loading the tissue segmentation prediction results from the first stage
        tissue = nib.load(os.path.join(self.images_path, patient_id, "pred_tissue.nii.gz")).get_fdata()
        tissue = process_label(tissue)
        T1 = None
        if os.path.exists(os.path.join(self.images_path, patient_id, "brain.nii.gz")):
            T1 = nib.load(os.path.join(self.images_path, patient_id, "brain.nii.gz")).get_fdata()
            T1 = process_image(T1)
        T2 = None
        if os.path.exists(os.path.join(self.images_path, patient_id, "T2-brain.nii.gz")):
            T2 = nib.load(os.path.join(self.images_path, patient_id, "T2-brain.nii.gz")).get_fdata()
            T2 = process_image(T2)
        three = None
        if os.path.exists(os.path.join(self.images_path, patient_id, "CT-brain.nii.gz")):
            three = nib.load(os.path.join(self.images_path, patient_id, "CT-brain.nii.gz")).get_fdata()
            three = process_image(three)
        elif os.path.exists(os.path.join(self.images_path, patient_id, "Flair-brain.nii.gz")):
            three = nib.load(os.path.join(self.images_path, patient_id, "Flair-brain.nii.gz")).get_fdata()
            three = process_image(three)
        elif os.path.exists(os.path.join(self.images_path, patient_id, "DWI-brain.nii.gz")):
            three = nib.load(os.path.join(self.images_path, patient_id, "DWI-brain.nii.gz")).get_fdata()
            three = process_image(three)
        PET = None
        pet_type = get_pet(caption.split(';')[2])
        if pet_type is not None:
            PET = nib.load(os.path.join(self.images_path, patient_id, pet_type + "-brain.nii.gz")).get_fdata()
            PET = process_image(PET)
        Random = None
        if os.path.exists(os.path.join(self.images_path, patient_id, "US-brain.nii.gz")):
            Random = nib.load(os.path.join(self.images_path, patient_id, "US-brain.nii.gz")).get_fdata()
            Random = process_image(Random)
        elif os.path.exists(os.path.join(self.images_path, patient_id, "PD-brain.nii.gz")):
            Random = nib.load(os.path.join(self.images_path, patient_id, "PD-brain.nii.gz")).get_fdata()
            Random = process_image(Random)
        elif os.path.exists(os.path.join(self.images_path, patient_id, "SWI-brain.nii.gz")):
            Random = nib.load(os.path.join(self.images_path, patient_id, "SWI-brain.nii.gz")).get_fdata()
            Random = process_image(Random)
        elif os.path.exists(os.path.join(self.images_path, patient_id, "T2s-brain.nii.gz")):
            Random = nib.load(os.path.join(self.images_path, patient_id, "T2s-brain.nii.gz")).get_fdata()
            Random = process_image(Random)
        # calculate edge map
        if T1 is not None:
            edge = sobel_3d(T1)
        elif T2 is not None:
            edge = sobel_3d(T2)
        elif three is not None:
            edge = sobel_3d(three)
        elif PET is not None:
            edge = sobel_3d(PET)
        elif Random is not None:
            edge = sobel_3d(Random)
        edge = process_image(edge.squeeze(0).numpy())
        # concat channels
        channels = []
        if T1 is not None:
            channels.append(T1)
        else:
            noise = torch.randn(tissue.shape)
            channels.append(noise)
        if T2 is not None:
            channels.append(T2)
        else:
            noise = torch.randn(tissue.shape)
            channels.append(noise)
        if three is not None:
            channels.append(three)
        else:
            noise = torch.randn(tissue.shape)
            channels.append(noise)
        if PET is not None:
            channels.append(PET)
        else:
            noise = torch.randn(tissue.shape)
            channels.append(noise)
        if Random is not None:
            channels.append(Random)
        else:
            noise = torch.randn(tissue.shape)
            channels.append(noise)
        channels.append(edge)

        image_dict = {
            "T1": channels[0],
            "T2": channels[1],
            "three": channels[2],
            "PET": channels[3],
            "Random": channels[4],
            "edge": channels[5],
            "tissue": tissue,
            "dk_struct": dk_struct
        }

        # Combine images into a tensor
        images = torch.cat([image_dict[key] for key in ["T1", "T2", "three", "PET", "Random", "edge", "tissue"]], dim=0)

        text = self.tokenizer(caption)[0]
        item = {
            'images': images,
            'dk-struct': image_dict['dk_struct'],
            'text': text,
            'affine': affine,
            'ID': patient_id,
        }

        return item


# Used for mutilmodal lesion inference
class LesionDataset(Dataset):
    def __init__(self, texts_path, images_path, index='Sheet2'):
        self.data = pd.read_excel(texts_path, sheet_name=index)
        self.images_path = images_path
        self.tokenizer, _ = load_text_encoder()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        caption = self.data.iloc[idx]['caption']
        patient_id = self.data.iloc[idx]['patient_id']
        patient_id = str(patient_id)
        T1 = None
        if os.path.exists(os.path.join(self.images_path, patient_id, "brain.nii.gz")):
            affine = nib.load(os.path.join(self.images_path, patient_id, "brain.nii.gz")).affine
            T1 = nib.load(os.path.join(self.images_path, patient_id, "brain.nii.gz")).get_fdata()
            T1 = process_image(T1)
        T2 = None
        if os.path.exists(os.path.join(self.images_path, patient_id, "T2-brain.nii.gz")):
            affine = nib.load(os.path.join(self.images_path, patient_id, "T2-brain.nii.gz")).affine
            T2 = nib.load(os.path.join(self.images_path, patient_id, "T2-brain.nii.gz")).get_fdata()
            T2 = process_image(T2)
        three = None
        if os.path.exists(os.path.join(self.images_path, patient_id, "CT-brain.nii.gz")):
            affine = nib.load(os.path.join(self.images_path, patient_id, "CT-brain.nii.gz")).affine
            three = nib.load(os.path.join(self.images_path, patient_id, "CT-brain.nii.gz")).get_fdata()
            three = process_image(three)
        elif os.path.exists(os.path.join(self.images_path, patient_id, "Flair-brain.nii.gz")):
            affine = nib.load(os.path.join(self.images_path, patient_id, "Flair-brain.nii.gz")).affine
            three = nib.load(os.path.join(self.images_path, patient_id, "Flair-brain.nii.gz")).get_fdata()
            three = process_image(three)
        elif os.path.exists(os.path.join(self.images_path, patient_id, "DWI-brain.nii.gz")):
            affine = nib.load(os.path.join(self.images_path, patient_id, "DWI-brain.nii.gz")).affine
            three = nib.load(os.path.join(self.images_path, patient_id, "DWI-brain.nii.gz")).get_fdata()
            three = process_image(three)
        PET = None
        pet_type = get_pet(caption.split(';')[2])
        if pet_type is not None:
            affine = nib.load(os.path.join(self.images_path, patient_id, pet_type + "-brain.nii.gz")).affine
            PET = nib.load(os.path.join(self.images_path, patient_id, pet_type + "-brain.nii.gz")).get_fdata()
            PET = process_image(PET)
        Random = None
        if os.path.exists(os.path.join(self.images_path, patient_id, "US-brain.nii.gz")):
            affine = nib.load(os.path.join(self.images_path, patient_id, "US-brain.nii.gz")).affine
            Random = nib.load(os.path.join(self.images_path, patient_id, "US-brain.nii.gz")).get_fdata()
            Random = process_image(Random)
        elif os.path.exists(os.path.join(self.images_path, patient_id, "PD-brain.nii.gz")):
            affine = nib.load(os.path.join(self.images_path, patient_id, "PD-brain.nii.gz")).affine
            Random = nib.load(os.path.join(self.images_path, patient_id, "PD-brain.nii.gz")).get_fdata()
            Random = process_image(Random)
        elif os.path.exists(os.path.join(self.images_path, patient_id, "SWI-brain.nii.gz")):
            affine = nib.load(os.path.join(self.images_path, patient_id, "WWI-brain.nii.gz")).affine
            Random = nib.load(os.path.join(self.images_path, patient_id, "SWI-brain.nii.gz")).get_fdata()
            Random = process_image(Random)
        elif os.path.exists(os.path.join(self.images_path, patient_id, "T2s-brain.nii.gz")):
            affine = nib.load(os.path.join(self.images_path, patient_id, "T2s-brain.nii.gz")).affine
            Random = nib.load(os.path.join(self.images_path, patient_id, "T2s-brain.nii.gz")).get_fdata()
            Random = process_image(Random)
        # calculate edge map
        if T1 is not None:
            edge = sobel_3d(T1)
        elif T2 is not None:
            edge = sobel_3d(T2)
        elif three is not None:
            edge = sobel_3d(three)
        elif PET is not None:
            edge = sobel_3d(PET)
        elif Random is not None:
            edge = sobel_3d(Random)
        edge = process_image(edge.squeeze(0).numpy())
        # concat channels
        channels = []
        if T1 is not None:
            channels.append(T1)
        else:
            noise = torch.randn((224, 256, 224))
            channels.append(noise)
        if T2 is not None:
            channels.append(T2)
        else:
            noise = torch.randn((224, 256, 224))
            channels.append(noise)
        if three is not None:
            channels.append(three)
        else:
            noise = torch.randn((224, 256, 224))
            channels.append(noise)
        if PET is not None:
            channels.append(PET)
        else:
            noise = torch.randn((224, 256, 224))
            channels.append(noise)
        if Random is not None:
            channels.append(Random)
        else:
            noise = torch.randn((224, 256, 224))
            channels.append(noise)
        channels.append(edge)

        image_dict = {
            "T1": channels[0],
            "T2": channels[1],
            "three": channels[2],
            "PET": channels[3],
            "Random": channels[4],
            "edge": channels[5]
        }

        # Combine images into a tensor
        images = torch.cat([image_dict[key] for key in ["T1", "T2", "three", "PET", "Random", "edge"]], dim=0)

        text = self.tokenizer(caption)[0]
        item = {
            'images': images,
            'text': text,
            'affine': affine,
            'ID': patient_id,
        }

        return item