# BrainSeg: A Generalized Framework for Comprehensive Multimodal Brain Tissue Segmentation, Parcellation, and Lesion Labeling
Official implementation code for BrainSeg. We proposes a novel AI-based tool for comprehensive brain imaging segmentation with generalizability across multiple modalities, including MRI, CT, PET, and ultrasound, as well as across the lifespan (from neonates to the elderly). This framework consists of three main components: B-Syn, B-CLIP, and BrainSeg.

***
## Model overview
<div style="text-align: center">
  <img src="figures/overview.png" width="100%" alt="BrainSeg Framework">
</div>

## Results
<div style="text-align: center">
  <img src="figures/results.png" width="100%" alt="Results">
</div>

***

***
# Get started with B-Syn and B-CLIP
## Step 1: Set up the environment for BiomedCLIP
Our B-CLIP fine-tunes BiomedCLIP text encoder based on LoRA, so you need to first configure the Biomedical environment: 

**1. First clone the latest BiomedCLIP model (the commit version we used is 27005c2, and earlier versions may have compatibility issues)**
```bash
cd /your/path/to/BiomedCLIP
git clone https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
```
**2. And then clone the latest BiomedBERT-abstract**
```bash
cd /your/path/to/BiomedBERT-abstract
git clone https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract
```
**3. Install the specific version of open_clip_torch**
```bash
pip install open_clip_torch==2.23.0 transformers==4.35.2 matplotlib
```
**4. To invoke your local path of Biomedical, you need to make a little modification to the source code of open-clip. Please follow: https://github.com/mlfoundations/open_clip/issues/772#issuecomment-1884355134**

**5. Finally modify the model configuration to enable the text encoder to output tokens. In /your/path/to/BiomedCLIP/open_clip_config.json, add the setting of output_tokens to the "text_cfg" dictionary**

**Before:**
```bash
   ...
   "context_length": 256
}
```
**After:**
```bash
    ...
    "context_length": 256,
    "output_tokens": true
}
```
**6. Now you can load the Biomedical model like this:**
```bash
model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
                         cache_dir='/your/path/to/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', 
                         cache_dir='/your/path/to/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
```
## Step 2: Preparation your data to train B-CLIP
You can organize your file directory as follows to train B-CLIP on your own data
```bash
prompt.xlsx                     # excel for text metadata
parameter.xlsx                  # excel for image parameter, used for data synthetic of B-Syn module
lesion_parameter.xlsx           # excel for lesion parameter, used for lesion synthetic of B-Syn module
data/
├── subject001/
│   ├── brain.nii.gz            # brain image
│   ├── tissue.nii.hz           # tissue map GT
│   ├── dk-struct.nii.gz        # roi map GT
│   ├── T2-brain.nii.gz         # T2 modality image, if have
│   ├── CT-brain.nii.gz         # CT modality image, if have
│   ├── ……                      # other modality image, if have
├── subject002 
├── subject003
└── ……
lesion/
├── subject01/
│   ├── brain.nii.gz            # brain image
│   ├── lesion.nii.hz           # lesion map GT
│   ├── Flair-brain.nii.gz      # T2-FLAIR modality image, if have
│   ├── T2-brain.nii.gz         # T2 modality image, if have
│   ├── ……                      # other modality image, if have
├── subject02 
├── subject03
└── ……
```
## Step 3: Train B-CLIP
Now you can start training B-CLIP. You can choose to train from scratch or load our pre-trained model of B-CLIP for fine-tuning. You can download our pretrained B-CLIP model through the following link: [BCLIP](https://drive.google.com/file/d/1yXsnsFRHFc_uZ84JoWh8wF63WGZjDdNh/view?usp=drive_link)
```bash
python /BCLIP/train.py  # Please change the path in the code to the path of your own data
```

***

***
# Get started with BrainSeg
## Step 1: Data prepocessing
Before starting training, we recommend that you preprocess the data. We suggest you use the same preprocessing steps as us, including registering all images to the MNI space and performing skull stripping. Then crop the image to (224, 256, 224). 
After preprocessing, your data directory should be structured to match the B-CLIP training format

## Step 2: Train BrainSeg
Now you can start training BrainSeg. You can choose to train from scratch or load our pre-trained model of BrainSeg for fine-tuning. You can download our pretrained BrainSeg model through the following link: [BrainSeg_tissue](https://drive.google.com/file/d/1oHgnOyCLNxjO3tn2iKG54-cyIEsKkVNS/view?usp=drive_link) for tissue segmentation, [BrainSeg_parc](https://drive.google.com/file/d/13Vl_3yaOgaWhUhdkS2IekR-sQnV4oCrA/view?usp=drive_link) for brain parcellation and [BrainSeg_lesion](https://drive.google.com/file/d/1qnw8pV1c6n0_kwUJx9iQXCJvDboFmxce/view?usp=drive_link) for lesion labeling
```bash
python train.py  # Please change the path in the code to the path of your own data
```

## Step 3: Inference using our pretrained model
We provide a set of example samples covering **diverse age groups, multiple modalities, and lesion cases** in [Sample](./Sample). You can run inference directly on these samples using our pre-trained model. You can also test on your own data, provided it is structured as follows:
```bash
test.xlxs                   # text metadata prompt
Sample/
├── sub001/
│   ├── brain.nii.gz        
│   ├── tissue.nii.hz        
│   ├── dk-struct.nii.gz     
│   ├── T2-brain.nii.gz            
│   ├── ……                  # any other modalities, if have                       
├── sub002/
│   ├── brain.nii.gz        
│   ├── tissue.nii.hz        
│   ├── dk-struct.nii.gz       
│   ├── CT-brain.nii.gz        
│   ├── ……                  # any other modalities, if have
├── sub003
└── ……
```
### 1. Tissue segmentation
You can run tissue segmentation inference using the following command:
```bash
python inference.py --texts_path ./test.xlsx --images_path ./Sample --img_size 224 256 224 --in_channels 6 --out_channels 4 --model_dir /your/path/for/BrainSeg_model --clip_dir /your/path/for/BCLIP --predir ./Sample --mode tissue --flag multi
```
*
    * `--texts_path`: Path to the Excel file (`.xlsx`) containing text prompts (Default: `./test.xlsx`)
    * `--images_path`: Directory path of the input images to be processed (Default: `./Sample`)
    * `--img_size`: Input image size as a tuple `(D, H, W)` (Default: `224 256 224`)
    * `--in_channels`: Number of input channels (Default: `6`)
    * `--out_channels`: Number of output classes (Default: `4`)
    * `--device`: Device to run the model on, e.g., `cuda` for GPU or `cpu` (Default: `cuda`)
    * `--model_dir`: Path to the directory containing your pretrained **BrainSeg** model
    * `--clip_dir`: Path to the directory containing your pretrained **B-CLIP** model
    * `--predir`: Directory path where the output results will be saved (Default: `./Sample`)
    * `--mode`: Specifies the prediction task (e.g., `tissue`, `dk` or `lesion`).
    * `--flag`: Determines the input modality strategy (Default: `multi`)
        * `multi`: Uses all modalities in the subject folder for fusion inference.
        * `single`: Uses a single modality for inference. Must be used with `--modality`.
    * `--modality`: Specifies the target modality filename when `--flag` is set to `single` (Default: `CT-brain.nii.gz`)

### 2. Brain Parcellation
You can run brain parcellation inference using the following command:
```bash
python inference.py --texts_path ./test.xlsx --images_path ./Sample --img_size 224 256 224 --in_channels 7 --out_channels 107 --model_dir /your/path/for/BrainSeg_model --clip_dir /your/path/for/BCLIP --predir ./Sample --mode dk
```

### 3. Lesion labeling
You can run lesion labeling inference using the following command:
```bash
python inference.py --texts_path ./test.xlsx --images_path ./Sample --img_size 224 256 224 --in_channels 6 --out_channels 4 --model_dir /your/path/for/BrainSeg_model --clip_dir /your/path/for/BCLIP --predir ./Sample --mode lesion
```

***

# Citation
If you find this work useful in your research, please cite:
> **Shijie Huang<sup>†</sup>, Zifeng Lian<sup>†</sup>, Dengqiang Jia<sup>†</sup>, Kaicong Sun<sup>†</sup>, Xiaoye Li<sup>†</sup>, Jiameng Liu<sup>†</sup>, Yulin Wang, Caiwen Jiang, Fangmei Zhu, Zhongxiang Ding, Han Zhang, Geng Chen<sup>&ast;</sup>, Feng Shi<sup>&ast;</sup>, Dinggang Shen<sup>&ast;</sup>. BrainSeg: A Generalized Framework for Comprehensive Multimodal Brain Tissue Segmentation, Parcellation, and Lesion Labeling. (Under Review)**

# [<font color=#F8B48F size=3>License</font> ](./LICENSE)
```shell
Copyright IDEA Lab, School of Biomedical Engineering, ShanghaiTech University, Shanghai, China.

Licensed under the the GPL (General Public License);
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Repo for BrainSeg: A Generalized Framework for Comprehensive Multimodal Brain Tissue Segmentation, Parcellation, and Lesion Labeling
Contact: huangshj@shanghaitech.edu.cn
         lianzf2024@shanghaitech.edu.cn
```
