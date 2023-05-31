# 10 minutes to run stable diffusion from scratch on Windows

> Downloading stable diffusion and related environment online can be very time-consuming. This note is intended to address this issue for quick testing on new device. If you want to test the latest resources (e.g. latest diffuser, onnxruntime etc.), it will not be a good reference.

1.	Download [all files](https://intel-my.sharepoint.com/:f:/r/personal/yaru_du_intel_com/Documents/Documents/report/2023Q2/AIGC-SD?csf=1&web=1&e=iWrkZS)

    | File Name    | Details | 
    |---------|-----|
    | Olive_SD1.5.zip   | onnx fp16 models for sd v1.5  | 
    | sdDML1.5.tar.gz     | packed conda enviroment including onxxruntime directml 1.15  | 
    | test_ml.py | a python scirpt to run sd model by using DirectML as backend  | 
    | latents_fp16.npy | latents data to get the same output image each time|
    | Anaconda3-2019.10-Windows-x86_64.exe | anaconda app  | 

2.	Click `Anaconda3***.exe` to install conda environment
3.	Unzip `sdDML1.5.tar.gz`
4.	Open anaconda prompt command and go to folder `sdDML1.5`
5.	Run the commands below  
    `.\Scripts\activate.bat`  <!-- activate the enviroment -->   

    `.\ Scripts \conda-unpack.exe` <!-- unpack the environment, only need to run at the very first time -->   
    >  

  	`conda list`  <!-- check the environment -->
    > Then, you can double check some key packages used for stable diffusion as below
    ```
    diffusers                 0.15.1                   pypi_0    pypi
    olive-ai                  0.1.0                    pypi_0    pypi
    onnx                      1.13.1                   pypi_0    pypi
    onnxruntime-directml      1.15.0                   pypi_0    psypi
    ```
    
6.	Go to the folder where has `test_dml.py`, and run the script  
`python test_dml.py`
    > Note, please point out the path where sd model is located in the scirpt before testing.


