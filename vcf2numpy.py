###Xiaoming Liu 11/03/2019
#This scirpt takes vcfs of individual windows and
#load them into numpy arrays
###
#Building dependencies

#os.chdir("/FYP_Liu")

import allel
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


#Initializing variables
headers = []
vcfs = glob.glob("/Users/nathanrobins/Documents/UG_proj/EDAR_Data_Splicing/*.vcf")
vcf_num = len(vcfs)

snps = [None]*vcf_num
colour_snps = [None]*vcf_num
pos = [None]*vcf_num
labels = [None] * vcf_num

plt.axis('off')

for i in range(vcf_num):
    headers.append(allel.read_vcf_headers(vcfs[i]))
#    labels[i] = vcfs[i].split('_')[-1]
    file = allel.read_vcf(vcfs[i])
    gt = allel.GenotypeArray(file['calldata/GT'])
    dim = gt.shape
    #print(gt[0,0,0])
    alt = file['variants/ALT']
    ref = file['variants/REF']

    #print(dim,alt.shape,ref.shape)
    #print(file['variants/numalt'])

    #print(alt)
#    allele_to_color ={'A':(255/2,255/2,0),'C':(0,255/2,255/2),'G':(255/2,0,255/2),'T':(255/2,255/2,255/2)}
#    INDEL = (255, 255, 255)

#    colour_gt = np.zeros(dim,dtype=np.dtype((np.float32,(1,3))))
#    for y in range(dim[0]): #for each snp
#        for x in range(dim[1]): #in each individual
#            for allele in range(2): #for each chromosome
#                gt_index = gt[y][x][allele]
#                if gt_index: #0,1,2... corresponding to ref and alternative alleles in that order
#                    nt = alt[y][gt_index-1] #-1 because starting with the first alternative allele (1-1=0 - first alt allele)
#                else:
#                    nt = ref[y]
#                channel = allele_to_color[nt] if len(nt)==1 else INDEL
#                #print(channel)
                #print(type(channel))
#                colour_gt[y][x][allele] = channel

    #print(gt.shape)
    #print(colour_gt.shape)
    #print(colour_gt)

    new_dim = (dim[0],dim[1]*dim[2],1)
    snps[i] = gt.reshape(new_dim)
    #colour_snps[i] = colour_gt.reshape((dim[0],dim[1]*dim[2],1),dtype = np.dtype((np.int32,(1,3))))
#    colour_snps[i] = colour_gt
    pos[i] = file['variants/POS']

    #plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    fig = plt.gcf()
    for ax in fig.axes:
        ax.axis('off')
        ax.margins(0, 0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
    plt.margins(0, 0)

    snp_im = gt.reshape((dim[0],dim[1]*dim[2]))
    plt.imshow(snp_im,cmap=cm.Greys_r)
#    plt.savefig('ImageData/Window_'+str(i))#bbox_inches = 'tight'

    #print('file finished')
    #print(len(headers[0][4]))

snps = np.asarray(snps)
pos = np.asarray(pos)

#print(snps.shape)
#print(snps)
#print(len(np.where(snps > 0)[-1])/(3445*5008))
#print(pos)
#print(pos.shape)

np.save('/Users/nathanrobins/Documents/UG_proj/NumpyData/bw_snps.npy',snps)
#np.save('NumpyData/colour_snps.npy',colour_snps)
np.save('/Users/nathanrobins/Documents/UG_proj/NumpyData/pos.npy',pos) #maybe use difference at the end?
np.save('/Users/nathanrobins/Documents/UG_proj/NumpyData/labels.npy',labels)
print(snps.shape)
print(snps[0].shape)
'''
fig = plt.figure()
ax1 = fig.add_subplot(121)
import matplotlib.cm as cm
dim = snps[0].shape
ax1.imshow(snps[0].reshape(dim[0],dim[1]),cmap=cm.Greys_r)
#print(colour_snps[0].shape)
ax1.imshow(colour_snps[0])
plt.show()
'''
