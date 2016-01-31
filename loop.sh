#!/bin/bash
 

while [ ! -f evoman_ended ]
do 
	python optimization_individualevolution_demo.py 
done

exit 0
	
