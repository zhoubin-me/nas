for x in `seq $1 $2`
do
    hbox-kill application_1515725804189_${x} &
done
