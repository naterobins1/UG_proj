#!/bin/bash

### Configuration for simulations

if [ "$#" -ne "4" ]; then
	echo "There needs to be 4 arguments"
	echo "arg1 = jar file"
	echo "arg2 = output file"
	echo "arg3 = demographic model"
	echo "arg4 = ??"
	exit 1;
fi

for x in {1..5}
do
	if [ ! -d "$2" ]; then 
	mkdir -p $2.$x;
	fi 
done


## 1) DEMOGRAPHIC MODEL

# The following parameters are taken from Marth et al. Genetics 2004
# 3 epoch: N3=10,000,N2=2,000,N1=20,000,T2=5,00,T1=3,000
# 1 epoch: N1=10,000
# 2 epoch: N2=10,000, N1=140,000, T1=2000

NRef=10000 # reference effective population size (since I am using the same Marth demo models as Matteo, we have the same NREF)

####### ------ 	CHECK WITH MATTEO WHAT THIS MEANS ----- ########
MARTH1='' # marth 1-epoch for CEU
MARTH2='-eN 0.05 1 -eN 0 14' # marth 2-epoch for CEU 
MARTH3='-eN 0.0875 1 -eN 0.075 0.2 -eN 0 2' # marth 3-epoch for CEU

# once the demographic model that is to be used is chosen, we must change the variable

if [ "$3" -eq "1" ]; then
	DEMO=$MARTH1;
fi
if [ "$3" -eq "2" ]; then
	DEMO=$MARTH2;
fi
if [ "$3" -eq "3" ]; then
	DEMO=$MARTH3;
fi


## 2) LOCUS & SAMPLE SIZE

Len=95000 # length of locus in bp
Theta=57 # mutation rate is 1.5e-8 per base per generation --> use formula: 4*NNREF*mut*Len
Rho=38 # recombinatio rate is 1e-8 --> formula as above

NChroms=128 # number of haplotypes (chrom) to extract 

####### -------  CHECK 'NChroms' with Matteo -----###########

## 3) SELECTION

SelPos=`bc <<< 'scale=2; 2694/95000'` # rel position of selected allele

Freq=`bc <<< 'scale=6; 1/100'` # frequency of the selected allele at the start of the selection --> there was a v low level of 370A therefore 0.01 is a low value??

#### ---- check all of the following, especially the seltime --> for my gene the time since fixation is 10740

if [ $4 == Binary ]; then 
    SelRange=`seq 0 100 400` # range and step for the selection coefficient to be estimated in 2*Ne units;
    NRepl=5000 # (20k) this is the number of replicates (simulations) per value of selection coefficient to be estimated; 
fi

if [ $4 == Continuous ]; then 
    SelRange=`seq 0 1 400` # range and step for the selection coefficient to be estimated in 2*Ne units;
    NRepl=250 # (250) this is the number of replicates (simulations) per value of selection coefficient to be estimated; 
fi

SelTime=`bc <<< 'scale=4; 440/40000'` # 11kya
# time for the start of selection in 4*Nref generations; e.g. 800/40000 is at 20kya, with Ne=10k and 25 years as gen time.

for i in {1..5}
do	
	for Sel in $SelRange
	do
		java -jar $1 -N $NRef -ms $NChroms $NRepl -t $Theta -r $Rho $Len -Sp $SelPos -SI $SelTime 1 $Freq -SAA $(($Sel * 2)) -SAa $Sel -Saa 0 -Smark $DEMO -thread 4 | gzip > $2.$i/msms..$Sel..$SelTime.txt.gz
	done
done
