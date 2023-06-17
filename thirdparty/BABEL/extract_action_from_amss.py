## A script to extract the sequences belonging to a specfic action from the AMASS database
## Takes as input th action label list
## Usage: python scripts/extract_action_from_amass.py --output_path .

import sys
import os
import joblib
import argparse
import subprocess
import json
import pandas as pd


def load_amass_db(db_dir, split='trainval'):
    if split == 'trainval':
        db_file = os.path.join(db_dir, 'amass_db_in_babel_trainval.pth.tar')
    if split == 'train':
        db_file = os.path.join(db_dir, 'amass_db_in_babel_train.pth.tar')
    if split == 'val':
        db_file = os.path.join(db_dir, 'amass_db_in_babel_val.pth.tar')
    db = joblib.load(db_file)
    return db

def load_babel_db(db_dir, split='trainval'):
    babel_db = os.path.join(db_dir, f'babel_v1.0_release_{split}.json')
    if split == 'trainval':
        print('Using BABEL {trainval} split for AMASS')
        babel_df = pd.read_json(babel_db)
    if split == 'train':
        print('Using BABEL {train} split for AMASS')
        babel_df = pd.read_json(babel_db)
    if split == 'val':
        print('Using BABEL {val} split for AMASS')
        babel_df = pd.read_json(babel_db)
    return babel_df

def main(args, vibe_fps=30):
    action_list = args.action_list
    amass_db = load_amass_db(args.db_dir, args.split)
    unique_seq_names = list(set(amass_db['amass_name']))
    babel_df = load_babel_db(args.db_dir, args.split)
    action_video_mds = {}
    for action in action_list:
        for seq_name in unique_seq_names:
            babel_url = babel_df.loc[babel_df['feat_p'] == seq_name, 'url'].iloc[0] 
            babel_md = babel_df.loc[babel_df['feat_p'] == seq_name, 'frame_ann'].iloc[0]
            if babel_md is not None and len(babel_md) != 0:
                babel_md = babel_md['labels']
                video_md = [{'seq_name': seq_name,
                             'act_cats': md_dict['act_cat'],
                             'start_frame': int(md_dict['start_t'] * vibe_fps),
                             'end_frame': int(md_dict['end_t'] * vibe_fps)} for md_dict in babel_md if action in md_dict['act_cat']]  # amass_fps=30
                
                if len(video_md) > 0:
                    video_md[0]['url'] = babel_url
                    video_md[0]['length'] = video_md[0]['end_frame'] - video_md[0]['start_frame']
                    action_video_mds.setdefault(action, []).append(video_md)
            else:
                # if no frame_ann, use seq_ann for all frames
                babel_md = babel_df.loc[babel_df['feat_p'] == seq_name, 'seq_ann'].iloc[0]['labels']
                dur = [babel_df.loc[babel_df['feat_p'] == seq_name, 'dur'].iloc[0]] * len(babel_md)
                video_md = [{'seq_name': seq_name,
                             'act_cats': md_dict['act_cat'],
                             'start_frame': 0,
                             'end_frame': int(d * vibe_fps)} for md_dict, d in zip(babel_md, dur) if action in md_dict['act_cat']]  # amass_fps=30
                
                if len(video_md) > 0:
                    video_md[0]['url'] = babel_url
                    video_md[0]['length'] = video_md[0]['end_frame'] - video_md[0]['start_frame']
                    action_video_mds.setdefault(action, []).append(video_md)

    # save the action_video_mds as json
    action_video_mds_json = os.path.join(args.output_path, f'action_video_mds_{args.split}.json')
    with open(action_video_mds_json, 'w') as f:
        json.dump(action_video_mds, f)
    print(f'Saved action_video_mds to {action_video_mds_json}')


## forwards movements + walk: https://babel-renders.s3.eu-central-1.amazonaws.com/007466.mp4
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract sequences from AMASS database')
    parser.add_argument('-a', '--action_list', help='Action list file', nargs="+",
                        default=["walk", "run"])
    parser.add_argument('-d', '--db_dir', help='AMASS database path',
                        default='/is/cluster/scratch/stripathi/pycharm_remote/VIBE/data')
    parser.add_argument('-s', '--split', help='Split of the database',
                        default='trainval',
                        choices=['trainval', 'train', 'val'])
    parser.add_argument('-o', '--output_path', help='Output path', required=True)
    args = parser.parse_args()

    main(args)
