from detector.mapper import MapperByUnproject

## Test 
if __name__ == "__main__":
    cam_para_file = "cam_para/MOT17/MOT17-02-SDP.txt"
    lookup_table = False 
    mapper = MapperByUnproject(cam_para_file, lookup_table, "MOT17")
    seq = cam_para_file.split("-")[-2]
    mapper.set_aicity_config("detector/config_mot17_"+seq+'.json')
    mapper.test_uv2xy()