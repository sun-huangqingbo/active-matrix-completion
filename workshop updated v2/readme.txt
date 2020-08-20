This folder contains the software and data used in

Huangqingbo Sun and Robert F. Murphy (2020) An Improved Matrix Completion Algorithm For Categorical Variables: Application to Active Learning of Drug Responses. Proceedings of the ICML 2020 Workshop on Real World Experiment Design and Active Learning. https://realworldml.github.io/files/cr/15_SunAndMurphyRevised.pdf 

Contacts: Huangqingbo Sun <huangqis@cs.cmu.edu> and Robert F. Murphy <murphy@cmu.edu>
	Computational Biology Department
	School of Computer Science
	Carnegie Mellon University

==================
The code requires the following Python packages

     numpy, pandas, matplotlib, sklearn, fancyimpute, seaborn

However, a modified version of the solver.py function from fancyimpute is required and is included in this folder.

To use it, copy it to the fancyimpute location (the command below does this).


cp solver.py $(pip show fancyimpute | grep Location | sed 's_Location:_ _;s_$_/fancyimpute/_')


(This command is included as a comment in main.sh)

==================
The file main.sh will run the analyses and generate the figures.


source main.sh

==================
