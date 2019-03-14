#!/bin/bash

for x in */
do
    cd "$x"
    vmd sample.gro sample.edr -e ps.vmd
    cd ../
done
