if [ "$1" == "-h" ]; then
  echo "Arg1: hostname"
  echo "Next X args GPU indicies"
  exit 0
fi

tmux new-session -d -s client
tmux send-keys -t client:0 "python client.py $1 --gpu_to_use $2" C-m

for(( i=5; i<=$#; i++ )); do
    tmux new-window -t client:$(($i - 4))
    tmux send-keys -t client:$(($i - 4)) "python caffe_client.py $1 --gpu_to_use ${!i}" C-m
done
