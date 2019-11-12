# PareidoliaCreator

Creates custom-made noise-pareidolic images using Python script, adjusted from the implementation of Perlin Noise for Python by Ruslan Grimov (https://github.com/ruslangrimov/perlin-noise-python-numpy). 

Six different 2D noise images generated using different node distances (5, 10, 50, 100, 250, and 500, image size = 1000x1000 pixels). 
These arere combined to a map for each image, which is then converted to grayscale after brightness augmentation. 
Each map is sharpened and eroded with a 9x9 kernel. 
Finally, a binary and otsy threshold is applied. 

Stimulus creation process:

![Stimulus creation process](https://github.com/AngelikaZa/PareidoliaCreator/blob/master/Pareidolia%20creation%20process.jpg)

Thresholded binarised images of faces can be superimposed to stimuli to recreate the Noise pareidolia test by Yoshiyuki Nishio (original paper: https://www.ncbi.nlm.nih.gov/pubmed/22649179)

Original pareidolia test can be found here: 
https://figshare.com/articles/The_noise_pareidolia_test/3187669
