#!/bin/bash
#

mkdir -p /media/ubuntu/ssd/T3/${2}/${1}

while :; do

    rsync --partial --timeout=20 -rlpgoDvz ${1}.sas.pvt:/home/ubuntu/data/*_header.json /media/ubuntu/ssd/T3/${2}/${1}/
    rsync --partial --timeout=20 -rlpgoDvz ${1}.sas.pvt:/home/ubuntu/data/*_data.out /media/ubuntu/ssd/T3/${2}/${1}/
    sleep 5
    
done

