import os, sys
from glob import glob
import random
random.seed(0)

root_dir = './T1/'
n_sub = 400

# all_subjects = sorted([f.split('-T1.nii.gz')[0] for f in os.listdir(root_dir)])[:n_sub]
all_subjects = sorted([f for f in os.listdir(root_dir)])[:n_sub]

# all_subjects1 = sorted(os.listdir(root_dir))[:n_sub]
# all_subjects_t1 = [fn.split('-T1.nii.gz')[0] for fn in all_subjects1]

# all_subjects2 = sorted(os.listdir(root_dir+'T2/'))[:n_sub]
# all_subjects_t2 = [fn.split('-T2.nii.gz')[0] for fn in all_subjects2]

# all_subjects = [fn for fn in all_subjects_t1 if fn in all_subjects_t2]

train_subs = all_subjects[:350]
val_subs = all_subjects[350:370]
test_subs = all_subjects[370:]
#save train-val-test subs in file
with open('./train_subs_mc.txt', 'w') as f:
	for sn in train_subs:
		f.write(sn+'\n')
with open('./val_subs_mc.txt', 'w') as f:
	for sn in val_subs:
		f.write(sn+'\n')
with open('./test_subs_mc.txt', 'w') as f:
	for sn in test_subs:
		f.write(sn+'\n')

# # coronal
# dim=0
# all_fnames = [
#     f for f in os.listdir('./multimodal_slices/t1/') if 'dim_{}'.format(dim) in f
# ]
# all_fnames = sorted(all_fnames)

# train_outf = './train_cor.txt'
# val_outf = './val_cor.txt'
# test_outf = './test_cor.txt'

# trainf = open(train_outf, 'w')
# valf = open(val_outf, 'w')
# testf = open(test_outf, 'w')

# for fname in all_fnames:
#     fsub = fname.split('_dim_')[0]
#     if fsub in val_subs:
#         valf.write(fname+'\n')
#     elif fsub in test_subs:
#         testf.write(fname+'\n')
#     else:
#         trainf.write(fname+'\n')
#     print(fname, 'done')

# trainf.close()
# valf.close()
# testf.close()


# axial
dim=1
all_fnames = [
    f for f in os.listdir('./mc_slices/t1/') if 'dim_{}'.format(dim) in f
]
random.shuffle(all_fnames)

train_outf = './train_axial_mc_t1.txt'
val_outf = './val_axial_mc_t1.txt'
test_outf = './test_axial_mc_t1.txt'

trainf = open(train_outf, 'w')
valf = open(val_outf, 'w')
testf = open(test_outf, 'w')

for fname in all_fnames:
    fsub = fname.split('_dim_')[0]
    if fsub in val_subs:
        valf.write(fname+'\n')
    elif fsub in test_subs:
        testf.write(fname+'\n')
    else:
        trainf.write(fname+'\n')
    print(fname, 'done')

trainf.close()
valf.close()
testf.close()

# dim=1
# all_fnames = [
#     f for f in os.listdir('./mc_slices/t1_l0/') if 'dim_{}'.format(dim) in f
# ]
# random.shuffle(all_fnames)

# train_outf = './train_axial_mc_t1_l0.txt'
# val_outf = './val_axial_mc_t1_l0.txt'
# test_outf = './test_axial_mc_t1_l0.txt'

# trainf = open(train_outf, 'w')
# valf = open(val_outf, 'w')
# testf = open(test_outf, 'w')

# for fname in all_fnames:
#     fsub = fname.split('_dim_')[0]
#     if fsub in val_subs:
#         valf.write(fname+'\n')
#     elif fsub in test_subs:
#         testf.write(fname+'\n')
#     else:
#         trainf.write(fname+'\n')
#     print(fname, 'done')

# trainf.close()
# valf.close()
# testf.close()

# # saggital
# dim=2
# all_fnames = [
#     f for f in os.listdir('./multimodal_slices/t1/') if 'dim_{}'.format(dim) in f
# ]
# all_fnames = sorted(all_fnames)

# train_outf = './train_sag.txt'
# val_outf = './val_sag.txt'
# test_outf = './test_sag.txt'

# trainf = open(train_outf, 'w')
# valf = open(val_outf, 'w')
# testf = open(test_outf, 'w')

# for fname in all_fnames:
#     fsub = fname.split('_dim_')[0]
#     if fsub in val_subs:
#         valf.write(fname+'\n')
#     elif fsub in test_subs:
#         testf.write(fname+'\n')
#     else:
#         trainf.write(fname+'\n')
#     print(fname, 'done')

# trainf.close()
# valf.close()
# testf.close()
