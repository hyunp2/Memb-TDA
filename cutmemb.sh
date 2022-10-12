#!/bin/bash

################
# 20220918
# Andres S. Arango
#

for NUMBER in {283..330}; do
	mkdir -p T.$NUMBER
	cd T.$NUMBER
	vmd -dispdev text -e ../slice_dcds.tcl -args $NUMBER
	cd ..
done
#done
#done
