export COMET_API_KEY="zooLWPj9P5NxLTjlEFJ7hvf5k"
export project_name="20240501"
yolo train model=Models/Base/yolov9c.pt imgsz=640 batch=32 epochs=150 data=Yaml/ScottLabelsV9.yaml project="yolov9" name="20240501_ScottlabelsV9_"
