## make sure you have the following libraries installed:
## - SimpleITK
## - Dicom
## - pandas
## - cv2
## - numpy

cd $HOME'/Desktop/AmanRana/Code'

mkdir $HOME'/Desktop/AmanRana/Datasets/LUNA'
mkdir $HOME'/Desktop/AmanRana/Datasets/Kaggle'
mkdir $HOME'/Desktop/AmanRana/Datasets/LUNA_processed'
mkdir $HOME'/Desktop/AmanRana/Datasets/LUNA_processed/Cancer'
mkdir $HOME'/Desktop/AmanRana/Datasets/LUNA_processed/Non Cancer'
mkdir $HOME'/Desktop/AmanRana/Datasets/Kaggle_processed'
mkdir $HOME'/Desktop/AmanRana/Datasets/Kaggle_processed/Cancer'
mkdir $HOME'/Desktop/AmanRana/Datasets/Kaggle_processed/Non Cancer'

#python process_LUNA_dataset.py $HOME'/kaggle_main/Data Science Bowl Kaggle/dataset/Annotated Lung Cancer Dataset/annotated_data' 'csv_files/candidates.csv' $HOME'/Desktop/AmanRana/Datasets/LUNA_processed'


python train_on_LUNA.py $HOME'/Desktop/AmanRana/Datasets/LUNA_processed' $HOME'/Desktop/AmanRana/Checkpoints/LUNA'

#python process_kaggle_dataset.py $HOME'/../../media/amanrana/GENERAL/stage1/' 'csv_files/stage1_labels.csv' $HOME'/Desktop/AmanRana/Datasets/Kaggle_processed' $HOME'/Desktop/AmanRana/Checkpoints/LUNA/Epoch_30/Model_checkpoint.ckpt-68400'

#python train_on_kaggle_dataset.py $HOME'/Desktop/AmanRana/Datasets/Kaggle_processed' $HOME'/Desktop/AmanRana/Checkpoints/Kaggle'

#python inference_submission.py <path/to/BEST/trained/model/on/kaggle/dataset> <path/for/test/dataset> $'/Desktop'
