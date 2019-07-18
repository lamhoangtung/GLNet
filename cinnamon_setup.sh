echo "Cloning code"
git clone https://github.com/lamhoangtung/ultra_high_resolution_segmentation
cd ultra_high_resolution_segmentation

echo "Downloading data"
mkdir data/
cd data/
drive clone 1Fxn1pjoJNRVNvEiDRTcLu8ZDItI3JUVS
unzip all_prj.zip

echo "Downloading previous weight"
cd ..
mkdir saved_models
mkdir experiments
cd experiments
drive clone 1VF44OnRG16su6i4zEdKsPYpnMP8mFRV5
unzip all_prj_stage_1_120_ep.zip