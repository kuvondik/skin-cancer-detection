# Skin Cancer Detection

This app identifies different types of skin lesions from dermoscopic images using an `EfficientNet-B0` model, which is lightweight yet highly accurate. It also provides explainability through `Grad-CAM` and `LIME`.
## Requirements

### To run on NVIDIA, install the CUDA build of PyTorch:
**For CUDA 12.1:**

`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`

**For CUDA 11.8:**

`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

### When installing PyTorch on macOS ARM (M1/M2/M3/M4), you must install using:
`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`

### Other packages that need to be installed:
`pip install numpy pandas scipy scikit-learn matplotlib seaborn tqdm Pillow scikit-image lime ipykernel jupyter`

## Dataset
If you use HAM10000 dataset, extract the dataset files into `src/data/HAM10000` folder and run `prepare_ham10000_folders.py` in `src/scripts` folder. It will prepare the dataset for training by creating folder structure for the data.

Structure:

`src/data/HAM10000_custom`

There will be 7 classified folders where specific images are kept:

- `/AKIEC`
- `/BCC`
- `/BKL`
- `/DF`
- `/MEL`
- `/NV`
- `/VASC`

## Steps to run the project
1. Prepare dataset
   1. If you want to use `HAM10000`, download the dataset. Here is the [link](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
   2. Extract the dataset files into `src/data/HAM10000` folder
   3. Run `prepare_ham10000_folders.py` in `src/scripts` folder
2. Run `skin_cancer_detector.py` 
