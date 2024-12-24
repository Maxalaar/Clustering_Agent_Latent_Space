PROJECTNAME=Clustering_Agent_Latent_Space

mkdir -p ~/packages/$PROJECTNAME

module load aidl/pytorch/2.2.0-cuda12.1
export PYTHONUSERBASE=~/packages/$PROJECTNAME
export PATH=$PATH:~/packages/$PROJECTNAME/

pip install --upgrade pip

pip install --user --no-cache-dir "ray[data,train,tune,serve]"==2.39.0
pip install --user --no-cache-dir dm_tree
pip install --user --no-cache-dir lz4
pip install --user --no-cache-dir tensorflow[and-cuda]
pip install --user --no-cache-dir tensorflow
pip install --user --no-cache-dir pygame
pip install --user --no-cache-dir GPUtil
pip install --user --no-cache-dir moviepy
pip install --user --no-cache-dir ffmpeg
pip install --user --no-cache-dir swig
pip install --user --no-cache-dir h5py
pip install --user --no-cache-dir filelock
pip install --user --no-cache-dir lightning
pip install --user --no-cache-dir --extra-index-url=https://pypi.nvidia.com "cudf-cu12==24.10.*" "dask-cudf-cu12==24.10.*" "cuml-cu12==24.10.*" "cugraph-cu12==24.10.*" "nx-cugraph-cu12==24.10.*" "cuspatial-cu12==24.10.*" "cuproj-cu12==24.10.*" "cuxfilter-cu12==24.10.*" "cucim-cu12==24.10.*" "pylibraft-cu12==24.10.*" "raft-dask-cu12==24.10.*" "cuvs-cu12==24.10.*" "nx-cugraph-cu12==24.10.*"
pip install --user --no-cache-dir bokeh
pip install --user --no-cache-dir selenium
pip install --user --no-cache-dir imbalanced-learn
pip install --user --no-cache-dir gymnasium==1.0.0
pip install --user --no-cache-dir Box2D
pip install --user --no-cache-dir gymnasium[mujoco]
pip install --user --no-cache-dir flappy-bird-gymnasium
pip install --user --no-cache-dir tetris-gymnasium
pip install --user --no-cache-dir ale-py