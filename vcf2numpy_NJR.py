#!/usr/bin/pytho

"""This script takes as an input .vcf files of indv windows & loads them into numpy arrays"""

## imports
import allel
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import glob

# create variables containing the vcfs
vcfs = glob.glob("/Users/nathanrobins/Documents/UG_proj/EDAR_Data_Splicing/*.vcf")
head = []
num_vcfs = len(vcfs)
position = [None]*num_vcfs
var = [None]*num_vcfs
label = [None]*num_vcfs

#plt.axis('off')

# for loop to seperate the vcfs
for n in range(num_vcfs):
	head.append(allel.read_vcf_headers(vcfs[n]))
	callset = allel.read_vcf(vcfs[n])
#	print(sorted(callset.keys()))
	# samples represents individuals
	# POS represents the position
	# calldata/GT = genotype calls
	GT = allel.GenotypeArray(callset['calldata/GT'])
	shape = GT.shape
#	print(shape)
	alt = callset['variants/ALT']
###### DOUBLE CHECK WITH MATTEO THAT I ONLY WANT TO TREAT THINGS AS BIALLELIC? --> 
###### AS THEN I CAN USE ...(callset, numbers={'ALT :1'})
#	print(alt)
	ref = callset['variants/REF']
#	print(ref)

	position = callset['variants/POS']	

#	print(shape)
#	new_shape = (shape[0],shape[1]*shape[2],1)
#	var[n] = GT.reshape(new_shape)
	
	im_shape = ((shape[0],shape[1]*shape[2]))

	# plotting
#	fig=plt.gcf()
#	for ax in fig.axes:
#        ax.axis('off')
#        ax.margins(0, 0)
#        ax.xaxis.set_major_locator(plt.NullLocator())
#        ax.yaxis.set_major_locator(plt.NullLocator())
#    plt.margins(0, 0)

	var_im = GT.reshape(im_shape)
	plt.imshow(var_im)
#	plt.savefig()

var = np.asarray(var)
position = np.asarray(position)

## save the data
np.save('/Users/nathanrobins/Documents/UG_proj/NumpyData/bw_snp.npy',var)
np.save('/Users/nathanrobins/Documents/UG_proj/NumpyData/positions.npy', position)
np.save('/Users/nathanrobins/Documents/UG_proj/NumpyData/labels.npy', label)

print(var.shape)
