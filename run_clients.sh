
for idn in {1..45..1}
do
    hadoop fs -rm -r /user/i-chenyunpeng/nas/client${idn}_logs &
done

for idn in {1..45..1}
do
    hbox-submit \
      --app-type "mxnet" \
      --files "train_cifar10.py,client.py,common,q_protocol.py,run_mxnet_cmd.py,net2sym.py,data" \
      --worker-num 1 \
      --worker-gpus 1 \
      --worker-memory 4024 \
      --hbox-cmd "python client.py --idn ${idn}" \
      --appName "nas_client" \
      --output "/user/i-chenyunpeng/nas/client${idn}_logs#logs" \
      --priority "VERY_HIGH" \
      --cacheFile "/user/i-chenyunpeng/nas/mxnet#mxnet" &
done
