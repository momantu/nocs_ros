# NOCS ROS 
The package wraps the [implementation of the paper Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation](https://github.com/hughw19/NOCS_CVPR2019.git) to a ROS service.


We use it in our Mobile Manipulation Tutorial. Our website is [here](https://github.com/momantu).

## Citation
```
 @InProceedings{Wang_2019_CVPR,
               author = {Wang, He and Sridhar, Srinath and Huang, Jingwei and Valentin, Julien and Song, Shuran and Guibas, Leonidas J.},
               title = {Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation},
               booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
               month = {June},
               year = {2019}
```


## Installation
Need to create a new ROS workspace, e.g., "nocs_ws". 
```
mkdir ~/nocs_ws
cd ~/nocs_ws
git clone https://github.com/momantu/nocs_ros.git src
```


Create conda environment and install dependencies for the [NOCS implementation](https://github.com/hughw19/NOCS_CVPR2019.git). 
```
conda create -n NOCS-env python=3.5 anaconda
conda activate NOCS-env
conda install setuptools
pip install -U --ignore-installed wrapt enum34 simplejson netaddr
pip install msgpack 
pip install tensorflow
pip install opencv-python
pip install --upgrade scikit-image
pip install open3d-python
pip install pycocotools
pip install keras
```
Install dependencies for running with ROS.
```
sudo apt update
sudo apt install python3-catkin-pkg-modules python3-rospkg-modules python3-empy		
sudo apt-get install python-catkin-tools python3-dev python3-numpy python3-yaml ros-melodic-cv-bridge
cd ~/nocs_ws/src/
git submodule update --init --recursive
cd ~/nocs_ws/
rosdep install --from-paths src --ignore-src -y -r
# replace 3.6 with 3.5 if you have /usr/include/python3.6m instead:
catkin_make --cmake-args \
            -DCMAKE_BUILD_TYPE=Release \
            -DPYTHON_EXECUTABLE=/usr/bin/python3 \
            -DPYTHON_INCLUDE_DIR=/usr/include/python3.5m \
            -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.5m.so
source devel/setup.zsh
pip install -U rosdep rosinstall_generator wstool rosinstall six
```

## Usage
### Pretrain weights
You can download the [checkpoints](http://download.cs.stanford.edu/orion/nocs/ckpts.zip) and store them under nocs_srv/logs/. We are using "logs/nocs_rcnn_res50_bin32.h5" as default.

### Parameter Setting
Set the parameters in the launch file ``pose_estimation_server.launch``:

`rgb_topic`: The topic name of the RGB image from your RGBD camera.

`dep_topic`: The topic name of the depth image, which is aligned to the RGB image, from your RGBD camera.

`camera_optical_frame`: The frame_id of the `rgb_topic`.

Fill the parameter `intrinsics_fx`, `intrinsics_fy`, `intrinsics_x0` and `intrinsics_y0` according to your camera's intrinsics matrix 
  ![image](https://github.com/momantu/nocs_ros/blob/master/images/intrinsics.gif)



`detect_range`: The range in which the objects should be detected (cm, default: 150).

### Start 6D Pose Estimation
Start the pose estimation ROS service:

```
conda activate NOCS-env
source ~/nocs_ws/devel/setup.bash
roslaunch nocs_srv pose_estimation_server.launch
```

Call the service with:
```
rosservice call /estimate_pose_nocs True
```
Then the estimated pose of the detected object will be published into the tf message as frame "/object_predicted".

## Known Issues

* Get ImportError of `import tensorflow.keras`:
    ```
    ImportError: Keras requires TensorFlow 2.2 or higher. Install TensorFlow via `pip install tensorflow`
    ```
  or get error:
    ```
    module 'tensorflow' has no attribute 'Session'
    ```
    or
    ```
    module 'tensorflow' has no attribute 'ConfigProto'
    ```
    Solution:
    Upgrade `tensorflow` to lastest version (>=2.2) to install `keras`, then downgrade `tensorflow` to required version (1.14.0):
    ```
    pip install --user --upgrade tensorflow-gpu
    pip install --user --upgrade tensorboard
    pip install keras==2.3.1
    pip install --user --upgrade tensorflow-gpu==1.14.0
    ```
    Reference: https://stackoverflow.com/questions/62465620/error-keras-requires-tensorflow-2-2-or-higher
    
* Get importerror: cannot import name 'pca' from 'matplotlib.mlab'. 
    It is because PAC() is removed in the version after Matplotlib 2.2.
    Solution:
    ```
    pip install matplotlib==2.2
    ```
    Reference: https://matplotlib.org/3.1.0/api/api_changes.html

* Get error:
    ```
    module 'scipy.misc' has no attribute 'imresize'
    ```
    Solution:
    ```
    pip install  --user --upgrade scipy==1.0.0
    ```
    
    If you get another error:
    ```
    No module named 'numpy.testing.decorators'
    ```
    Solution:
    ```
    pip install numpy==1.18
    pip install scipy==1.1.0
    pip install scikit-learn==0.21.3
    ```
      
    Reference: https://stackoverflow.com/questions/59474533/modulenotfounderror-no-module-named-numpy-testing-nosetester
  
* If you receive these messages:
    ```
    2020-05-06 14:06:31.448663: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
    2020-05-06 14:06:31.471415: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 3600000000 Hz
    2020-05-06 14:06:31.471798: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5627feacbed0 executing computations on platform Host. Devices:
    2020-05-06 14:06:31.471820: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (0): <undefined>, <undefined>
    2020-05-06 14:06:31.891868: W tensorflow/compiler/jit/mark_for_compilation_pass.cc:1412] (One-time warning): Not using XLA:CPU for cluster because envvar TF_XLA_FLAGS=--tf_xla_cpu_global_jit was not set.  If you want XLA:CPU, either set that envvar, or use experimental_jit_scope to enable XLA:CPU.  To confirm that XLA is active, pass --vmodule=xla_compilation_cache=1 (as a proper command-line flag, not via TF_XLA_FLAGS) or set the envvar XLA_FLAGS=--xla_hlo_profile.
    ```
    Ignore this warning by adding these line in the code:
    ```
    # Just disables the warning, doesn't enable AVX/FMA
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    ```
    Or in terminal, set:
    ```
    export TF_CPP_MIN_LOG_LEVEL=2
    ```
    Reference: https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u


