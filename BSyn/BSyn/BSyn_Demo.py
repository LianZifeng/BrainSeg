import argparse
import os
import pandas as pd
import random
import SimpleITK as sitk
from syn import *


class ModalitySynthesizer:
    def __init__(self):
        # -------------------------
        # paths & constants
        # -------------------------
        self.age_threshold = 60
        self.root_path = "../OASIS3_OAS30013_ses-d0102"

        # -------------------------
        # load meta tables
        # -------------------------
        self.meta_table = pd.read_excel("../parameter_example.xlsx")
        self.meta_table_tumor = pd.read_excel("../lesion_parameter_example.xlsx")

        self.modality_dict = self._build_modality_dict(self.meta_table)
        self.modality_dict_tumor = self._build_modality_dict(self.meta_table_tumor)

        # -------------------------
        # preload images
        # -------------------------
        self.brain_img = sitk.ReadImage(os.path.join(self.root_path, "brain.nii.gz"))
        self.tissue_img = sitk.ReadImage(os.path.join(self.root_path, "tissue.nii.gz"))
        self.roi_img = sitk.ReadImage(os.path.join(self.root_path, "dk-struct.nii.gz"))

        self.brain_data = sitk.GetArrayFromImage(self.brain_img)
        self.tissue_data = sitk.GetArrayFromImage(self.tissue_img)
        self.roi_data = sitk.GetArrayFromImage(self.roi_img)

        self.pet_prefixes = [
            "AV45-brain.nii.gz", "FDG-brain.nii.gz", "Dynamic-brain.nii.gz",
            "PIB-brain.nii.gz", "SUV-brain.nii.gz"
        ]

    def _build_modality_dict(self, table):
        d = {}
        for m in table.iloc[:, 2].unique():
            d[m] = {"filename": [], "age": []}
        for _, row in table.iterrows():
            d[row[2]]["filename"].append(row[0])
            d[row[2]]["age"].append(row[1])
        return d

    def synthesize(self, modality_key: str, lesion_type=None):
        """
        The ONLY public API.
        """
        # -------------------------
        # CT
        # -------------------------
        if modality_key == "CT-brain.nii.gz":
            out, _ = GMM_ct(
                self.tissue_data, self.roi_data,
                self._random_age(modality_key),
                self.age_threshold, self.meta_table,
                modality_key
            )
            return self._wrap_output(out)

        # -------------------------
        # PET
        # -------------------------
        if modality_key in self.pet_prefixes:
            return self._wrap_output(
                GMM_pet(self.roi_data, self.brain_data,
                        self._random_age(modality_key),
                        self.age_threshold, self.meta_table)
            )

        # -------------------------
        # lesion modalities
        # -------------------------
        if lesion_type!=None:
            return self._synthesize_lesion(modality_key,lesion_type)

        # -------------------------
        # MRI
        # -------------------------
        return self._synthesize_mri(modality_key)

    def _synthesize_mri(self, modality_key):
        filename = "OASIS3_OAS30013_ses-d0102"
        age=69

        if modality_key == "brain.nii.gz":
            out = GMM_mri_t1(self.brain_data, self.roi_data, self.tissue_data,
                             age, self.age_threshold, self.meta_table,
                             filename, modality_key)

        elif modality_key == "T2-brain.nii.gz":
            out = GMM_mri_t2(self.brain_data, self.roi_data, self.tissue_data,
                             age, self.age_threshold, self.meta_table,
                             filename, modality_key)

        elif modality_key in ["DWI-brain.nii.gz", "Flair-brain.nii.gz"]:
            out, self.tissue_out = GMM_DWI_flair(
                self.brain_data, self.roi_data, self.tissue_data,
                age, self.age_threshold, self.meta_table,
                filename, modality_key
            )

        elif modality_key in ["PD-brain.nii.gz", "T2s-brain.nii.gz"]:
            out = GMM_mri_t2(self.brain_data, self.roi_data, self.tissue_data,
                             age, self.age_threshold, self.meta_table,
                             filename, modality_key)

        elif modality_key == "SWI-brain.nii.gz":
            out = GMM_mri_t1(self.brain_data, self.roi_data, self.tissue_data,
                             age, self.age_threshold, self.meta_table,
                             filename, modality_key)

        elif modality_key in ["US-brain.nii.gz"]:
            out = GMM_pet(self.roi_data, self.brain_data,
                          age, self.age_threshold, self.meta_table)

        elif modality_key in self.pet_prefixes:
            out = GMM_pet(self.roi_data, self.brain_data,
                          age, self.age_threshold, self.meta_table)

        else:
            out = GMM_random(self.roi_data, age,
                             self.age_threshold, self.meta_table)

        return self._wrap_output(out)

    def _synthesize_lesion(self, modality_key, lesion_type):
        # lesion_type = "tumor" if "tumor" in modality_key else "stroke"
        example = f"{lesion_type}.nii.gz"

        root = "../UKBiobank_4655123" if lesion_type == "tumor" else "../UKBiobank2_4652340"
        lesion_path = f"../{lesion_type}_example"

        brain_img = sitk.ReadImage(os.path.join(root, modality_key))
        brain_data = sitk.GetArrayFromImage(brain_img)
        tissue_img = sitk.ReadImage(os.path.join(root, "tissue.nii.gz"))
        tissue_data = sitk.GetArrayFromImage(tissue_img)
        roi_img = sitk.ReadImage(os.path.join(root, "dk-struct.nii.gz"))
        roi_data = sitk.GetArrayFromImage(roi_img)

        out, self.roi_out, self.tissue_out = real_DWI_flair_lesion(
            brain_data, roi_data, tissue_data,
            self.meta_table_tumor,
            lesion_path, modality_key,
            os.path.join(lesion_path, example)
        )

        return self._wrap_output(out, brain_img), self._wrap_output(self.tissue_out[0,0], brain_img)

    def _random_age(self, modality_key):
        return random.choice(self.modality_dict[modality_key]["age"])

    def _wrap_output(self, tensor, ref_img=None):
        if ref_img is None:
            ref_img = self.brain_img
        img = sitk.GetImageFromArray(tensor.detach().cpu().numpy())
        img.CopyInformation(ref_img)
        return img


def get_args():
    parser = argparse.ArgumentParser(
        description="Modality Synthesizer (MRI / PET / CT / Lesion)"
    )

    parser.add_argument(
        "--modality",
        type=str,
        required=True,
        choices=[
            "brain.nii.gz",
            "T2-brain.nii.gz",
            "DWI-brain.nii.gz",
            "Flair-brain.nii.gz",
            "PD-brain.nii.gz",
            "SWI-brain.nii.gz",
            "T2s-brain.nii.gz",
            "US-brain.nii.gz",
            "CT-brain.nii.gz",
            "AV45-brain.nii.gz",
            "FDG-brain.nii.gz",
            "Dynamic-brain.nii.gz",
            "PIB-brain.nii.gz",
            "SUV-brain.nii.gz",
        ],
        help="Modality to synthesize"
    )

    parser.add_argument(
        "--lesion_type",
        type=str,
        default=None,
        choices=[None, "tumor", "stroke"],
        help="Lesion type (optional)"
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="./outputs",
        help="Output directory"
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default="GMM",
        help="Output filename prefix"
    )

    parser.add_argument(
        "--save_tissue",
        action="store_true",
        help="Save tissue output if exists (lesion case)"
    )

    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    syn = ModalitySynthesizer()

    # -------------------------
    # normal synthesis
    # -------------------------
    if args.lesion_type is None:
        out = syn.synthesize(args.modality)
        out_path = os.path.join(
            args.out_dir, f"{args.prefix}_{args.modality}"
        )
        sitk.WriteImage(out, out_path)

        print(f"[✓] Saved modality to {out_path}")

    # -------------------------
    # lesion synthesis
    # -------------------------
    else:
        out, out_tissue = syn.synthesize(
            args.modality,
            lesion_type=args.lesion_type
        )

        out_path = os.path.join(
            args.out_dir, f"{args.prefix}_{args.modality}"
        )
        sitk.WriteImage(out, out_path)

        if args.save_tissue:
            tissue_path = os.path.join(
                args.out_dir, f"{args.prefix}_tissue_{args.modality}"
            )
            sitk.WriteImage(out_tissue, tissue_path)

        print(f"[✓] Saved lesion modality to {out_path}")


if __name__ == "__main__":
    main()