for x in `seq $1 $2`
do
    hbox-kill application_1506323180152_${x} &
done
