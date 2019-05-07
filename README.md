# phase-separation

<img src="https://user-images.githubusercontent.com/25011342/57324525-957b4080-70cd-11e9-847b-9fc9748e6ba4.png" width="100px" height="100px" />

### A package to determine clusters in simulations of mixtures.  Some of the work based off of: https://github.com/upandacross/Hetrogenous-vs-Homogenous

This package contains functions in Python that allow a user to determine the amount
of components of a fluid in solution from images rendered from VMD. To do so, the
following workflow should be used:

1. Render images in VMD: An example VMD script is given, which will loop through a
   set of simulation trajectories and render an image of each system.

2. Process the images: `image_process.py` contains a variety of functions that will
   process the raw image from VMD.  Processing techniques include:

   * Image convolution
   * Otsu's Thresholding
   * K-Mean's Clustering to determine dominant image color

3. Connected Components labeling: `connections.py` contains `count_connections`
   function that will count the number of components from a processed image.
