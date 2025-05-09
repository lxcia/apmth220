## Git workflow

1. `git clone https://github.com/kevinli5941/spatialGNN.git` locally.
2. **Anytime you begin a work session for this project, please remember to `git pull` so we avoid merge conflicts!!!**
3. Commit and push your changes frequently! No need to do branching. Just `git add .` -> `git commit -m [message]` -> `git push`
4. The first time you push, make sure to do `git push --set_upstream origin master`! Every push after that can just be `git push`

## SIGEL Documentation 

SIGEL documentation and tutorials can be found [here](https://zipging.github.io/SIGEL.github.io/?#0-welcome-to-sigel).

## Environment setup

`conda create -n spatialgnn python=3.11`

`conda activate spatialgnn`

`cd SIGEL`

If you're using the O2 computer cluster, load the gcc module with: `module load gcc/6.2.0` (gcc/9.2.0 doesn't work apparently)

`python3 setup.py install`

## Download data

Download data from [here](https://drive.google.com/drive/folders/1C3Gk-HVYp2dQh4id8H68M9p8IWEOIut_) and put all the data under `spatialGNN/SIGEL/data/`, as specified in Part 1.3 in the SIGEL documentation. **DON'T put it in the `spatialGNN/data` folder!**

## Important notes

* When doing anything from the SIGEL tutorial, do everything within the `spatialGNN/SIGEL` directoy as the working directory, or some of the relative imports won't work properly. 

* When working with DGL on O2, make sure to install `dgl=1.1.3` or else it won't work with the GLIBC version on O2.
