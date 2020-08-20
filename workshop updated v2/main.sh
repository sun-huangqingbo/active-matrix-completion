#pip install scipy==1.4.1
#pip install h5py==2.10.0
#pip install fancyimpute
#cp solver.py $(pip show fancyimpute | grep Location | sed 's_Location:_ _;s_$_/fancyimpute/_')
python runanalyses.py
python reproduce_figs.py
