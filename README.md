# YDATA Parallel Tempering Meetup

# Setting up the environment
The example uses `tensorflow==1.14.0` and Python=3.6. 
Set up an environment with Anaconda:

1. Download and install Anaconda by following instructions for Mac/Linux [here](https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart) and for Windows [here](https://problemsolvingwithpython.com/01-Orientation/01.03-Installing-Anaconda-on-Windows/)

2. Open terminal/cmd and set up the environment
```bash
conda create -n ydata-pt-env python=3.6
conda activate ydata-pt-env
conda install pip jupyterlab sklearn
```
3. Install tensorflow:

Windows/Mac:
```bash
pip install tensorflow==1.14.0
```

On Linux:
```bash
$(which pip) install tensorflow==1.14.0
```

4. Clone git repository:

On Mac/Linux
```bash
cd ~
git clone https://github.com/vlad-user/ydata-parallel-tempering.git
cd ydata-parallel-tempering
```

On Windows
```cmd
cd %HOMEPATH%
git clone https://github.com/vlad-user/ydata-parallel-tempering.git
cd ydata-parallel-tempering
```

5. Run `jupyterlab`:

On Windows/Mac
```bash
jupyter lab
```
On Linux:
```bash
$(which jupyter) lab
```