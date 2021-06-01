# periorbitAI
AI system to segment and measure periorbital features from color photographs

In this repo you will find a python script (run_periorbitAI.py) to segment a color photograph (in JPG format) and calculate the periorbital measures described in https://doi.org/10.1016/j.ajo.2021.05.007.

It takes two inputs a root directory name and a directory name containing all photographs:

python run_periorbitAI.py /data/periorbitAI photos

It will create:
1) a directory "periorbitAI_figures" where it will put a segmetation overlay and a report with periorbital measures
2) a csv (periorbitAI_measures.csv) with the measures in mm for each subject (subject IDs are determined by JPG names)

We provide an example photograph in "photos" and the output for this photo in "example_output."


