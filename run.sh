## Modifying these two variables to replace the experimental dataset.
## DATASET is the abbreviation for dataset in our project and DATASETNAME is the folder name of the dataset.
# DATASET=luad
# DATASETNAME=LUAD-HistoSeg
DATASET=bcss
DATASETNAME=BCSS-WSSS


# Run enhance_cam
# python enhance_cam.py                           \
#     --dataset $DATASET                          \
#     --trainroot datasets/$DATASETNAME/train/    \
#     --testroot datasets/$DATASETNAME/test/      \
#     --max_epoches 25                            \
#     --batch_size 64 \
#    >> logs/stage1_on_$DATASET.txt

# Run mask_seg
python mask_seg.py                              \
    --dataroot datasets/$DATASETNAME            \
    --dataset $DATASET                          \
    --weights checkpoints/stage1_checkpoint_trained_on_$DATASET.pth\
    --epochs 30                                 \
    --batch-size 64                             \
  >> logs/stage2_on_$DATASET.txt
