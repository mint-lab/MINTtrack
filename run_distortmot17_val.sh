
# python run_distortmot17_val.py --seq MOT17-02 --fps 30 --wx 0.1 --wy 0.2 --a 40 --cdt 30  --hp 
# python run_distortmot17_val.py --seq MOT17-04 --fps 30 --wx 0.5 --wy 0.5 --a 10 --cdt 30  --hp 
# python run_distortmot17_val.py --seq MOT17-05 --fps 14 --wx 0.1 --wy 3.0 --a 10 --cdt 10  --hp 
# python run_distortmot17_val.py --seq MOT17-09 --fps 30 --wx 0.5 --wy 1.0 --a 10 --cdt 30  --hp 
# python run_distortmot17_val.py --seq MOT17-10 --fps 30 --wx 5.0 --wy 5.0 --a 60 --cdt 10  --hp 
# python run_distortmot17_val.py --seq MOT17-11 --fps 30 --wx 5.0 --wy 5.0 --a 40 --cdt 10  --hp 
# python run_distortmot17_val.py --seq MOT17-13 --fps 25 --wx 5.0 --wy 5.0 --a 40 --cdt 10  --hp 
# python eval_distortmot17.py > distortmot17_UCMCTrack.log

python run_distortmot17_val.py --seq MOT17-02 --fps 30 --wx 0.1 --wy 0.2 --a 40 --cdt 30  --flag_unpro --hp 
python run_distortmot17_val.py --seq MOT17-04 --fps 30 --wx 0.1 --wy 0.2 --a 10 --cdt 30  --flag_unpro --hp 
python run_distortmot17_val.py --seq MOT17-05 --fps 14 --wx 0.1 --wy 0.2 --a 10 --cdt 30  --flag_unpro --hp 
python run_distortmot17_val.py --seq MOT17-09 --fps 30 --wx 0.1 --wy 0.2 --a 10 --cdt 30  --flag_unpro --hp 
python run_distortmot17_val.py --seq MOT17-10 --fps 30 --wx 0.1 --wy 0.2 --a 60 --cdt 30  --flag_unpro --hp 
python run_distortmot17_val.py --seq MOT17-11 --fps 30 --wx 0.1 --wy 0.2 --a 40 --cdt 30  --flag_unpro --hp 
python run_distortmot17_val.py --seq MOT17-13 --fps 25 --wx 0.1 --wy 0.2 --a 40 --cdt 30  --flag_unpro --hp 
python eval_distortmot17.py > distortmot17_UCMCTrack+AICity_new.log

python run_distortmot17_val.py --seq MOT17-02 --fps 30 --wx 0.1 --wy 0.2 --a 40 --cdt 30 --lookup_table --flag_unpro --hp 
python run_distortmot17_val.py --seq MOT17-04 --fps 30 --wx 0.1 --wy 0.2 --a 10 --cdt 30 --lookup_table --flag_unpro --hp 
python run_distortmot17_val.py --seq MOT17-05 --fps 14 --wx 0.1 --wy 0.2 --a 10 --cdt 30 --lookup_table --flag_unpro --hp 
python run_distortmot17_val.py --seq MOT17-09 --fps 30 --wx 0.1 --wy 0.2 --a 10 --cdt 30 --lookup_table --flag_unpro --hp 
python run_distortmot17_val.py --seq MOT17-10 --fps 30 --wx 0.1 --wy 0.2 --a 60 --cdt 30 --lookup_table --flag_unpro --hp 
python run_distortmot17_val.py --seq MOT17-11 --fps 30 --wx 0.1 --wy 0.2 --a 40 --cdt 30 --lookup_table --flag_unpro --hp 
python run_distortmot17_val.py --seq MOT17-13 --fps 25 --wx 0.1 --wy 0.2 --a 40 --cdt 30 --lookup_table --flag_unpro --hp 
python eval_distortmot17.py > distortmot17_UCMCTrack+AICity+Lookup_new.log

pause