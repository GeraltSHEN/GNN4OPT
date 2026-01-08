# Src
https://github.com/ds4dm/learn2branch/tree/master

# SCIP solver

Set-up a desired installation path for SCIP / SoPlex (e.g., `/opt/scip`):
```
export SCIPOPTDIR='/opt/scip'
```

## SoPlex

SoPlex 4.0.1 (free for academic uses)

https://soplex.zib.de/download.php?fname=soplex-4.0.1.tgz

what I used:
git clone -b bugfix-40 --single-branch https://github.com/scipopt/soplex.git

```
tar -xzf soplex-4.0.1.tgz
cd soplex-4.0.1/
mkdir build
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=$SCIPOPTDIR
make -C ./build -j 4
make -C ./build install
cd ..
```

# SCIP

SCIP 6.0.1 (free for academic uses)

https://scip.zib.de/download.php?fname=scip-6.0.1.tgz

```
tar -xf scip-6.0.1.tar
cd scip-6.0.1/
```

Apply patch file in `learn2branch/scip_patch/`

```
patch -p1 < $RCAC_SCRATCH/GNN4OPT/scip_patch/vanillafullstrong.patch
```

```
mkdir build
cmake -S . -B build -DSOPLEX_DIR=$SCIPOPTDIR -DCMAKE_INSTALL_PREFIX=$SCIPOPTDIR
make -C ./build -j 4
make -C ./build install
cd ..
```

For reference, original installation instructions [here](http://scip.zib.de/doc/html/CMAKE.php).

## Cython

Required to compile PySCIPOpt and PySVMRank
```
conda install "cython<3"
```

## PySCIPOpt

SCIP's python interface (modified version)

```
pip install git+https://github.com/ds4dm/PySCIPOpt.git@ml-branching
```


### Set Covering
```
# Generate MILP instances
python 01_generate_instances.py setcover
# Generate supervised learning datasets
python 02_generate_samples.py setcover -j 4  # number of available CPUs
```


### Combinatorial Auction
```
# Generate MILP instances
python 01_generate_instances.py cauctions
# Generate supervised learning datasets
python 02_generate_samples.py cauctions -j 4  # number of available CPUs
```

### Capacitated Facility Location
```
# Generate MILP instances
python 01_generate_instances.py facilities
# Generate supervised learning datasets
python 02_generate_samples.py facilities -j 4  # number of available CPUs
```

### Maximum Independent Set
```
# Generate MILP instances
python 01_generate_instances.py indset
# Generate supervised learning datasets
python 02_generate_samples.py indset -j 4  # number of available CPUs
```