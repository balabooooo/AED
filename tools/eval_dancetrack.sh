# usage: bash eval_dancetrack.sh [GT path] [split txt path] [result_txt path] [output path]
# e.g. 
# bash eval_sportsmot.sh \
# /data/Datasets/SportsMOT/dataset/val \
# /data/Datasets/SportsMOT/splits_txt/val.txt \
# exps/sportsmot_infer_results/infer1/result_txt \
# exps/sportsmot_infer_results/infer1

python3 ./TrackEval/scripts/run_mot_challenge.py \
    --SPLIT_TO_EVAL val \
    --METRICS HOTA CLEAR Identity \
    --GT_FOLDER $1 \
    --SEQMAP_FILE $2 \
    --SKIP_SPLIT_FOL True \
    --TRACKERS_TO_EVAL '' \
    --TRACKER_SUB_FOLDER '' \
    --USE_PARALLEL True \
    --NUM_PARALLEL_CORES 8 \
    --PLOT_CURVES False \
    --TRACKERS_FOLDER $3 |& tee -a $4/score.log