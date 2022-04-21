#!/bin/bash

# template sh file

for tau_tv in 1 2 3 4
do
	for tau_l1 in 1 2 3 4
	do
		for obj_idx in 1 2 3
		do
			qsub -v tau_tv=$tau_tv,tau_l1=$tau_l1,obj_idx=$obj_idx -N tv${tau_tv}_l1${tau_l1}_obj${obj_idx} -o tv${tau_tv}_l1${tau_l1}_obj${obj_idx}  cm2_3D_deconv_obj_select_qsub.qsub
		done
	done
done

