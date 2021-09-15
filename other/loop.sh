#!/bin/bash

experiment_name='individual_demo'

while [ ! -f $experiment_name"/neuroended" ]
do 
	python optimization_specialist_demo.py
done

exit 0
	
