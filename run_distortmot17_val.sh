python run_distortmot17_val.py --seq MOT17-02 --fps 30 --wx 0.1 --wy 0.2 --a 40 --cdt 30  --hp 
python eval_distortmot17.py > distortmot17_UCMCTrack.log

python run_distortmot17_val.py --seq MOT17-02 --fps 30 --wx 0.1 --wy 0.2 --a 40 --cdt 30  --flag_unpro --hp 
python eval_distortmot17.py > distortmot17_UCMCTrack+AICity.log

python run_distortmot17_val.py --seq MOT17-02 --fps 30 --wx 0.1 --wy 0.2 --a 40 --cdt 30 --lookup_table --flag_unpro --hp 
python eval_distortmot17.py > distortmot17_UCMCTrack+AICity+Lookup.log

pause