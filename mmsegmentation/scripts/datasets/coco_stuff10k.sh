# download
mkdir -p data/coco_stuff10k && cd data/coco_stuff10k
wget http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/cocostuff-10k-v1.1.zip

# unzip
unzip cocostuff-10k-v1.1.zip

# --nproc means 8 process for conversion, which could be omitted as well.
python /mmsegmentation/tools/convert_datasets/coco_stuff10k.py ./ --nproc 8
