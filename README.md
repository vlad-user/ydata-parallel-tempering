# YDATA Parallel Tempering Meetup

# Setting up the environment
The example uses `tensorflow==1.14.0` and Python=3.6. 
Set up an environment with Anaconda:

1. Download and install Anaconda following these instructions for [Mac/Linux](https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart) and for [Windows](https://problemsolvingwithpython.com/01-Orientation/01.03-Installing-Anaconda-on-Windows/)
2. Open terminal/cmd and set up the environment
```bash
conda create -n ydata-pt-env python=3.6
conda activate ydata-pt-env
conda install pip jupyterlab sklearn
```
Following on Windows/Mac:
```bash
pip install tensorflow==1.14.0
```
On Linux:
```bash
$(which pip) install tensorflow==1.14.0
```
3. Clone git repository:
```bash
cd ~
git clone https://github.com/vlad-user/ydata-parallel-tempering.git
cd ydata-parallel-tempering
```
4. Run `jupyterlab`:
On Windows/Mac
```bash
jupyter lab
```
On Linux:
```bash
$(which jupyter) lab
```