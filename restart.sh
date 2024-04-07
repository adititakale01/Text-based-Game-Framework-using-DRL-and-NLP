#!/bin/bash
echo "Removing required files"
rm train*.txt
rm saver*.txt
rm check_student_*.txt
rm *.pdf
rm tsne.txt
rm -r Savednetworks
rm -rf StudentSavednetworks
rm -rf MDQNSavednetworks
rm -rf logs
rm *pyc
rm checkStates*
rm evalQval*
