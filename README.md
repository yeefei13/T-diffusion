
# Multi Model Effictive DM

Multimodel Q-diffusion is a post quantization project inpired by Q-Diffusion(https://github.com/Xiuyu-Li/q-diffusion/tree/master)

## Overview

This project aims to develop a set of methods that improve the accuracy of the Q-Diffusion model while quantizing the model to reduce the model size. This involves separately quantizing and calibrating the model in order to better capture local data patterns, then inference using different models at different sections of timesteps. Our hypothesis is that quantizing and calibrating a model on a more local set of timestep data could make the model inference better at the timesteps that it's quantized and calibrated on, compared to the model being quantized and calibrated over the whole timestep. Therefore, having different models quantized and calibrated over different segments of time step data and inference using multiple models could potentially produce better results.


### Installation

Clone this repository, and then create and activate a suitable conda environment named `qdiff` by using the following command:

```bash
conda env create -f environment.yml
conda activate qdiff
```

### Usage
This project focus only on the CIFAR-10 dataset using DDIM model, to due to loading multiple different model, I change the code to encode different model's file path within in the script/sample_diffusion_ddim.py file
```bash
# CIFAR-10 (DDIM)

python scripts/sample_diffusion_ddim.py --config configs/cifar10.yml --use_pretrained --timesteps 100 --eta 0 --skip_type quad --ptq --weight_bit 4 --quant_mode qdiff --cali_st 20 --cali_batch_size 32 --cali_n 128 --quant_act --act_bit 8 --a_sym --split --cali_iters 1000 --resume n/a --cali_data_path cifar_sd1236_sample2048_allst.pt -l <outpt_folder>

```

### Calibration
To conduct the calibration process, you must first generate the corresponding calibration datasets. I used the Q-Diffusion provided example calibration datasets [here](https://drive.google.com/drive/folders/12TVeziKWNz_HmTAIxQLDZlHE33PKdpb1?usp=sharing), specifically the cifar_sd1236_sample2048_allst.pt dataset. These datasets contain around 1000-2000 samples of intermediate outputs at each time step. 

To reproduce the calibrated checkpoints, you can use the following commands:

```bash
# CIFAR-10 (DDIM)
python scripts/sample_diffusion_ddim.py --config configs/cifar10.yml --use_pretrained --timesteps 100 --eta 0 --skip_type quad --ptq --weight_bit 4 --quant_mode qdiff --cali_st 20 --cali_batch_size 32 --cali_n 128 --quant_act --act_bit 8 --a_sym --split --cali_iters 1000 --cali_data_path cifar_sd1236_sample2048_allst.pt -l <outpt_folder>
```
I used [torch-fidelity](https://github.com/toshas/torch-fidelity) for IS and FID computation.
