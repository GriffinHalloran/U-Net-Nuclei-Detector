# U-Net-Nuclei-Detector

This is code for a Nuclei Detector that used a U-net neural network.

Nuclei detection and segmentation in digital microscopic tissue images has been a long standing challenge in the field of histology as well as computer vision. While there have been many attempts to solve this problem, especially using recent advancements in machine learning and deep learning, such methods pose a fundamental problem of lack of enough properly annotated data. Hence, we shift the focus to non-machine learning methods in computer vision to address this issue. We investigate two non-machine learning methods ( Hough circle detection and Felzenszwalb and Huttenlocher segmentation) and two machine learning methods ( Mask R-CNNs and UNets) for nuclei segmentation. While our main focus is on non-machine learning methods, we use machine learning methods to create baselines for our experiments. The models are compared using mean IOU scores over the dataset.


You do not need much to run this code. You will just need to have a few modules downloaded, like tensorflow, tqdm, scikit-image, and keras. Along with this, you will need to download the data yourself because it was too large to be uploaded. The images can be found here:

https://www.kaggle.com/c/data-science-bowl-2018

If they are put in the same directory as this, then the program unet.py will work. Just run python unet.py! 
The code will iterate through the images, run them through some preprocessing, then define a network and run the data through it. Unfortunately, this takes quite a bit of time, so instead I’ve included the model of a network already trained using it. So instead of training an entire network, you can just see how it preforms on new data. It will take five random images from the test data and attempt to segment them. They are then shown one after the other so that you can compare them to each other.
