from detector.object_draw_predict import save_lookup_table


if __name__ == '__main__':
    # save_lookup_table('detector/config_mot17_02.json', image_step=10, save_prefix = "detector/data/")
    # save_lookup_table('detector/config_mot17_04.json', image_step=10, save_prefix = "detector/data/")
    save_lookup_table('detector/config_mot17_05.json', image_size=(640, 480),image_step=5, save_prefix = "detector/data/")
    # save_lookup_table('detector/config_mot17_09.json', image_step=10, save_prefix = "detector/data/")
    save_lookup_table('detector/config_mot17_10.json', image_step=10, save_prefix = "detector/data/")
    save_lookup_table('detector/config_mot17_11.json', image_step=10, save_prefix = "detector/data/")
    save_lookup_table('detector/config_mot17_13.json', image_step=10, save_prefix = "detector/data/")