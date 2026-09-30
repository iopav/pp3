\# BDNN for Tactile Classification



This repository contains the implementation of \*\*BDNN\*\* (Brain-inspired Developmental Neural Network) for tactile texture classification, based on the arXiv paper:



> Xing, J., Chen, L., Zhang, Z., Hasan, M. N., \& Zhang, Z. B. (2024). \*An Intrinsically Knowledge-Transferring Developmental Spiking Neural Network for Tactile Classification\*. arXiv preprint arXiv:2410.00745. \[arXiv link](https://arxiv.org/abs/2410.00745)



---



\## 📦 Requirements



Install the required Python packages:



```bash

pip install -r requirements.txt

---Dataset

This code deploys BDNN on a public texture dataset:



Lima B M R, Danyamraju V N S S, de Oliveira T E A, et al. A multimodal tactile dataset for dynamic texture classification. Data in Brief, 2023, 50: 109590.



To test the adaptive ability of BDNN, the dataset has been divided into two sub-datasets.
If the folder "data" is empty, you can download the dataset here.

https://drive.google.com/file/d/13j3mVNFdb_iVZTB1Uku9IxjAVhsP6XNy/view?usp=drive_link



After the downloading, replace the original subfolders in the folder "data" with the downloaded dataset.



---Running Experiments

There are two main options:



1\. Directly learn 6 classes with BDNN

bash

Copy code

python main_exp1.py

2\. Learn 6 new classes based on a pretrained model for previous 6 classes

bash

Copy code

python main_exp2.py

-- Notes

Make sure the data folder is correctly structured before running the experiments.



It is recommended to create a virtual environment using Python 3.6, which makes it easier to ensure that the requirements.txt can be installed and run correctly. 



The scripts will automatically handle training, evaluation, and logging of results.



References

Xing, J., Chen, L., Zhang, Z., Hasan, M. N., \& Zhang, Z. B. (2024). An Intrinsically Knowledge-Transferring Developmental Spiking Neural Network for Tactile Classification. arXiv:2410.00745.



Lima B M R, Danyamraju V N S S, de Oliveira T E A, et al. A multimodal tactile dataset for dynamic texture classification. Data in Brief, 2023, 50: 109590.

