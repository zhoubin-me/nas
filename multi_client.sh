if [ "$1" == "-h" ]; then
  echo "Arg1: hostname"
  echo "Next X args GPU indicies"
  exit 0
fi

tmux new-session -d -s client
#tmux send-keys -t client:0 "python client.py $1 $2" C-m

for(( i=2; i<=$#; i++ )); do
    tmux new-window -t client:$(($i))
    tmux send-keys -t client:$(($i)) "export PATH=/home/i-chenyunpeng/zhoubin/miniconda3/bin:$PATH; python client.py $1 ${!i}" C-m
done
