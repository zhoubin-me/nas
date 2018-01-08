hadoop fs -rm -r /user/i-chenyunpeng/nas/logs
hbox-submit \
  --app-type "tensorflow" \
  --files "main.py,policy.py,q_protocol.py,server.py" \
  --worker-num 1 \
  --worker-gpus 0 \
  --worker-memory 40240 \
  --hbox-cmd "python main.py" \
  --appName "nas_server" \
  --output "/user/i-chenyunpeng/nas/logs#logs" \
  --priority "VERY_HIGH" \
  --cacheFile "/user/i-chenyunpeng/nas/torch#torch"
