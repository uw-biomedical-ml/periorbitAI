# periorbitAI
AI system to segment and measure periorbital features from color photographs

In this repo you will find a python script (run_periorbitAI.py) to segment a color photograph (in JPG format) and calculate the periorbital measures described in https://doi.org/10.1016/j.ajo.2021.05.007.

It takes two inputs a root directory name and a directory name containing all photographs:

python run_periorbitAI.py /data/periorbitAI photos

![test](https://user-images.githubusercontent.com/25671442/120400553-ef718300-c2f2-11eb-91ce-c5ad1d68415b.JPG)

It will create:
1) a directory "periorbitAI_figures" where it will put a segmetation overlay and a report with periorbital measures
![Screen Shot 2021-06-01 at 4 09 15 PM](https://user-images.githubusercontent.com/25671442/120400982-d87f6080-c2f3-11eb-9f0e-2ffd92416484.png)

![test_report](https://user-images.githubusercontent.com/25671442/120400617-0adc8e00-c2f3-11eb-9053-f6deb3bd2462.png)

2) a csv (periorbitAI_measures.csv) with the measures in mm for each subject (subject IDs are determined by JPG names)
![Screen Shot 2021-06-01 at 4 05 43 PM](https://user-images.githubusercontent.com/25671442/120400749-60189f80-c2f3-11eb-997e-e951faf88b8e.png)


We provide an example photograph in "photos" and the output for this photo in "example_output," shown above.

You must install the following Python packages:

Pytorch (1.4.0)
See installation instructions.

Matplotlib
cv2
skimage
scipy

conda install --file requirements.txt

or

pip install -r requirements.txt

If having troubles with import cv2, try:

conda install --channel https://conda.anaconda.org/menpo opencv3

Installation
git clone https://github.com/uw-biomedical-ml/irf-periorbitAI




