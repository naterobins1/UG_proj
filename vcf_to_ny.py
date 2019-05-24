#!/bin/python


import allel
import numpy as np
from numpy import newaxis
import scipy
import pandas
import matplotlib.pyplot as plt
import skimage

#read the vcf file and only extract those with 1 alternative allele
callset = allel.read_vcf('/home/nathanrobins/UG_proj/EDAR_Data_Splicing/2.109510927-109605828.ALL.chr2.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes-CEU.vcf', numbers={'ALT': 1,'AF': 1})

#callset = allel.read_vcf('/home/shiyun/Documents/sandbox/Data/RNF207.vcf', fields=['ALT', 'AF'], alt_number=1)
#This will somehow leaves only 
#callset = allel.read_vcf('/home/shiyun/Documents/sandbox/Data/RNF207.vcf')

#show the fields
#sorted(callset.keys())

#get the genotype
gt = allel.GenotypeArray(callset['calldata/GT'])
#reshape it so that all genotypes (0/0,0/1,1/1) shows along rows. Notice that the row is haplotype while column is SNP sites.
dim = gt.shape
dimnew = (dim[0],dim[1]*dim[2],1)
#remove rows which are all 0 values (means no polymorphism) caused by DataSlicer
snps = gt.reshape(dimnew)
r = np.copy(snps)
r1 = np.squeeze(r, axis=2)
r2 = r1[~np.all(r1 == 0, axis=1)]
r3 = r2.transpose()
snpsnew = r3[:, :, newaxis]

#Remove sites whose minor allele frequency is below the set threshold.
idx = np.where(np.mean(snpsnew[:,:,0], axis=0) >= 0.01)[0]
snpsnew = snpsnew[:,idx,:] 

def sort(data, ordering):
        """
        Sort rows and/or columns given an ordering.

        Keyword Arguments:
            ordering: either 'rows_freq', 'cols_freq', 'rows_dist', 'cols_dist'

        Returns:
            0
        """
        if ordering == 'rows_freq':
            uniques, counts = np.unique(data, return_counts=True, axis=0)
            counter = 0
            for j in counts.argsort()[::-1]:
                #argsort() used to return the indices that would sort an array.
                #[::-1] from end to first
                for z in range(counts[j]):
                    data[counter,:,:] = uniques[j,:,:]
                    counter += 1
        elif ordering == 'cols_freq':
            uniques, counts = np.unique(data, return_counts=True, axis=1)
            counter = 0 #
            for j in counts.argsort()[::-1]:
                for z in range(counts[j]):
                    data[:,counter,:] = uniques[:,j,:]
                    counter += 1
        elif ordering == 'rows_dist':
            uniques, counts = np.unique(data, return_counts=True, axis=0)
            # most frequent row in float
            top = uniques[counts.argsort()[::-1][0]].transpose().astype('float32')
            # distances from most frequent row
            distances = np.mean(np.abs(uniques[:,:,0] - top), axis=1)
            # fill in from top to bottom
            counter = 0
            for j in distances.argsort():
                for z in range(counts[j]):
                    data[counter,:,:] = uniques[j,:,:]
                    counter += 1
        elif ordering == 'cols_dist':
            uniques, counts = np.unique(data, return_counts=True, axis=1)
            # most frequent column
            top = uniques[:,counts.argsort()[::-1][0]].astype('float32')
            # distances from most frequent column
            distances = np.mean(np.abs(uniques[:,:,0] - top), axis=0)
            # fill in from left to right
            counter = 0
            for j in distances.argsort():
                for z in range(counts[j]):
                    data[:,counter,:] = uniques[:,j,:]
                    counter += 1

sort(snpsnew,'rows_freq') 
sort(snpsnew,'cols_freq')

image = np.copy(snpsnew[:,:,0])
data = np.zeros((128, 128, 1), dtype='uint8')
data[:,:,0] = skimage.transform.resize(image, (128,128), anti_aliasing=True, mode='reflect').astype('uint8')

#snpsnew = snpsnew[newaxis, :, :, :]
print(data.shape)

np.save('/home/nathanrobins/UG_proj/NumpyData/CEU#2.npy',data)