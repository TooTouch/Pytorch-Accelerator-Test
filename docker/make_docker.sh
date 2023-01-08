nvidia-docker run -it -h accelerator \
        -p 1291:1291 \
        --ipc=host \
        --name accelerator \
        -v /ssd2:/projects \
	-v /hdd/datasets:/datasets \
	nvcr.io/nvidia/pytorch:22.12-py3 bash
