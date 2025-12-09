import os
import pandas as pd
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn import functional as F
from testencoder import load_text_encoder
import random
import re
from monai.transforms import Compose, NormalizeIntensityd
from syn import *

# Downsample the image by half to save the GPU memory.
def process_image(image):
    image = image[None, None, :, :, :]
    downsample_image = torch.tensor(image).float()
    downsample_image = F.interpolate(downsample_image, scale_factor=(0.5, 0.5, 0.5), mode='trilinear', align_corners=False)
    image_dict = {"image": downsample_image}
    transforms = Compose([NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True)])
    image_dict = transforms(image_dict)
    image = image_dict['image'].squeeze(0)
    return image


def process_label(label):
    label = label[None, None, :, :, :]
    downsampled_label = torch.tensor(label).float()
    downsampled_label = F.interpolate(downsampled_label, scale_factor=(0.5, 0.5, 0.5), mode='nearest').long()
    label = downsampled_label.squeeze(0)
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


def get_age(caption):
    age_match = re.search(r'Age:\s*([\d.]+)\s*(years?|months?|gestational\s+weeks?)', caption)
    if age_match:
        age_value = float(age_match.group(1))
        unit = age_match.group(2).lower()
        if 'year' in unit:
            return age_value
        elif 'month' in unit:
            return age_value / 12
        elif 'gestational' in unit:
            return age_value / 52

"""
Args:
    texts_path: path for text meatdata
    images_path: path for images
    parameter_path: path for parameter of real images, used for data synthesis
    lesion_path: path for lesion images
    lesion_parameter_path: path for parameter of lesion images, used for lesion data synthesis

the format of the excel of texts_path is as follows:
There are two columns in total, namely patient_id and caption
patient_id stores the name of the subject (folder), 
and caption stores the text prompt, such as age, gender, modality and other information

"""

class CLIPDataset(Dataset):
    def __init__(self, texts_path, images_path, parameter_path, lesion_path, lesion_parameter_path):
        self.data = pd.read_excel(texts_path)
        self.images_path = images_path
        self.tokenizer, _ = load_text_encoder()
        if parameter_path is not None:
            self.meta_table = pd.read_excel(parameter_path)
        if lesion_path is not None:
            self.lesion_path = lesion_path
        if lesion_parameter_path is not None:
            self.meta_table_lesion = pd.read_excel(lesion_parameter_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        caption = self.data.iloc[idx]['caption']
        patient_id = self.data.iloc[idx]['patient_id']
        dk_struct = nib.load(os.path.join(self.images_path, patient_id, "tissue.nii.gz")).get_fdata()
        tissue = nib.load(os.path.join(self.images_path, patient_id, "tissue.nii.gz")).get_fdata()
        # the first channel for T1 image
        T1 = None
        if os.path.exists(os.path.join(self.images_path, patient_id, "brain.nii.gz")):
            T1 = nib.load(os.path.join(self.images_path, patient_id, "brain.nii.gz")).get_fdata()
        # the second channel for T2 image
        T2 = None
        if os.path.exists(os.path.join(self.images_path, patient_id, "T2-brain.nii.gz")):
            T2 = nib.load(os.path.join(self.images_path, patient_id, "T2-brain.nii.gz")).get_fdata()
            T2 = process_image(T2)
        # the fourth channel for PET
        PET = None
        pet_type = get_pet(caption.split(';')[2])
        if pet_type is not None:
            PET = nib.load(os.path.join(self.images_path, patient_id, pet_type + "-brain.nii.gz")).get_fdata()
            PET = process_image(PET)
        # The fifth channel includes Ultrasound, PD, SWI, and T2star. If there are real images, real images will be used. 
        # If there are no real images, 80% will be replaced by random noise and 20% by synthetic data
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
        if Random is None:
            if T1 is None:
                Random = torch.randn(1, 112, 128, 112)
            else:
                prob = torch.rand(1).item()
                if prob < 0.8:
                    Random = torch.randn(1, 112, 128, 112)
                else:
                    flag = random.choice(["US-brain.nii.gz", "PD-brain.nii.gz", "SWI-brain.nii.gz", "T2s-brain.nii.gz", 'brain.nii.gz'])
                    age = get_age(caption)
                    if flag == "T2s-brain.nii.gz" or flag == "PD-brain.nii.gz":
                        Random = GMM_mri_t2(T1, dk_struct, tissue, age, 55, self.meta_table, patient_id, flag)
                        Random = process_image(Random.cpu().numpy())
                    elif flag == "SWI-brain.nii.gz":
                        Random = GMM_mri_t1(T1, dk_struct, tissue, age, 45, self.meta_table, patient_id, flag)
                        Random = process_image(Random.cpu().numpy())
                    elif flag == "US-brain.nii.gz":
                        Random = GMM_pet(dk_struct, T1, age, 100, self.meta_table)
                        Random = process_image(Random.cpu().numpy())
                    else:
                        Random = GMM_random(dk_struct, 20, age, self.meta_table)
                        Random = process_image(Random.cpu().numpy())
                    # rewrite the text prompt
                    pattern = r'Modality: \[(.*?)\]'
                    match = re.search(pattern, caption)
                    if match:
                        modality_list = match.group(1).split(', ')
                        modality_list[-1] = 'Random'
                        new_modality = f"Modality: [{', '.join(modality_list)}]"
                        caption = re.sub(pattern, new_modality, caption)
        # The third channel includes CT, T2 flair, and DWI. If there are real images, real images will be used. 
        # One is diretly to use normal images, and the other is to randomly attach synthetic lesions to the real images
        Lesion = None
        if os.path.exists(os.path.join(self.images_path, patient_id, "CT-brain.nii.gz")):
            Lesion = nib.load(os.path.join(self.images_path, patient_id, "CT-brain.nii.gz")).get_fdata()
            modality_key = "CT-brain.nii.gz"
        elif os.path.exists(os.path.join(self.images_path, patient_id, "Flair-brain.nii.gz")):
            Lesion = nib.load(os.path.join(self.images_path, patient_id, "Flair-brain.nii.gz")).get_fdata()
            modality_key = "Flair-brain.nii.gz"
        elif os.path.exists(os.path.join(self.images_path, patient_id, "DWI-brain.nii.gz")):
            Lesion = nib.load(os.path.join(self.images_path, patient_id, "DWI-brain.nii.gz")).get_fdata()
            modality_key = "DWI-brain.nii.gz"
        if Lesion is not None:
            prob = torch.rand(1).item()
            if prob < 0.4:
                Lesion = process_image(Lesion)
            else:
                age = get_age(caption)
                if modality_key == "CT-brain.nii.gz":
                    Lesion, dk_struct, tissue = real_ct_lesion(Lesion, dk_struct, tissue, age, 20, self.meta_table, self.meta_table_lesion, modality_key, self.lesion_path)
                else:
                    Lesion, dk_struct, tissue = real_DWI_flair_lesion(Lesion, dk_struct, tissue, age, 20, self.meta_table, patient_id, self.meta_table_lesion, self.lesion_path, modality_key)
                tissue = tissue.squeeze().cpu().numpy()
                Lesion = process_image(Lesion).cpu()
        
                T1 = None
                T2 = None
                PET = None
                Random = torch.randn(1, 112, 128, 112)
        
                pattern = r'Modality: \[(.*?)\]'
                match = re.search(pattern, caption)
                if match:
                    modality_list = match.group(1).split(', ')
                    modality_list = ['None' if i != 2 else modality_list[i] for i in range(len(modality_list))]
                    new_modality = f"Modality: [{', '.join(modality_list)}]"
                    caption = re.sub(pattern, new_modality, caption)
        
                caption += "; Diagnosis: Lesion"
        else:
            Lesion = torch.randn(1, 112, 128, 112)
        tissue = process_label(tissue)
        channels = []
        if T1 is not None:
            T1 = process_image(T1)
            channels.append(T1)
        else:
            noise = torch.randn(tissue.shape)
            channels.append(noise)
        if T2 is not None:
            channels.append(T2)
        else:
            noise = torch.randn(tissue.shape)
            channels.append(noise)
        # channels.append(Lesion)
        if Lesion is not None:
            Lesion = process_image(Lesion)
            channels.append(Lesion)
        else:
            noise = torch.randn(tissue.shape)
            channels.append(noise)
        if PET is not None:
            channels.append(PET)
        else:
            noise = torch.randn(tissue.shape)
            channels.append(noise)
        # channels.append(Random)
        if Random is not None:
            channels.append(Random)
        else:
            noise = torch.randn(tissue.shape)
            channels.append(noise)
        channels.append(tissue)
        images = torch.cat(channels, dim=0)

        text = self.tokenizer(caption)[0]
        item = {
            'images': images,
            'text': text,
            'caption': caption
        }
        return item