#!/usr/bin/env bash
 
while read requirement; do conda install --yes -n capstone $requirement; done < ../requirements.txt
