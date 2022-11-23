IMAGE_NAME=mmseg
NOW=`date +"%Y%m%d%I%M%S"`
DATA_DIR=/mnt/data1/ozora/data
DATASET=coco_stuff10k
MODEL=lraspp_m-v3-d8
GPU=1

build:
	docker build -t ${IMAGE_NAME} .

train:
	docker run --rm -it --name ${IMAGE_NAME}-${USER}-${NOW} \
			            -e WANDB_API_KEY=${WANDB_API_KEY} \
			            -e ROOT_DIR=${ROOT_DIR} \
			            -e EPOCHS=${EPOCHS} \
						-v ${PWD}/src:/opt/src \
						-v ${PWD}/configs:/opt/configs \
						-v ${DATA_DIR}:/opt/data \
						--shm-size=4g \
						--gpus=all \
						${IMAGE_NAME} bash



train:
	docker run --rm -it --name ${IMAGE_NAME}-${USER}-${NOW} \
			            -e WANDB_API_KEY=${WANDB_API_KEY} \
			            -e ROOT_DIR=${ROOT_DIR} \
			            -e EPOCHS=${EPOCHS} \
						-v ${PWD}/src:/opt/src \
						-v ${PWD}/configs:/opt/configs \
						-v ${DATA_DIR}:/opt/data \
						-v ${PWD}/results:/mmsegmentation/results \
						--shm-size=4g \
						--gpus=all \
						${IMAGE_NAME} sh -c "cp /opt/src/hooks/* /mmsegmentation/mmseg/core/hook && python tools/train.py /opt/configs/${DATASET}/${MODEL}.py --work-dir results/${DATASET}/${MODEL}/ --gpu-id ${GPU}"
