import sys
sys.path.append("./")
from util.evaluation import teta_eval
import json
import os.path as osp
import os
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('evalutaion for tao dataset', add_help=False)
    parser.add_argument('--ann_file', default='', type=str)
    parser.add_argument('--res_path', default='', type=str)
    parser.add_argument('--show_pre_video', default=False, type=bool,
                        help='Result will be shown for each video if True, otherwise, the overall result will be shown.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args_parser()
    ann_file = args.ann_file
    res_path = args.res_path
    res_file = os.path.basename(res_path)
    res_path = os.path.dirname(res_path)

    if args.show_pre_video == False:
        teta_eval(ann_file, res_path, res_file)
    else:
        ann = json.load(open(ann_file, 'r'))
        res = json.load(open(osp.join(res_path, res_file), 'r'))
        v_ids = [v['id'] for v in ann['videos']]
        for v_id in v_ids:
            print(f'video id: {v_id}')
            ann_i = {}
            for k,v in ann.items():
                if k == 'tracks':
                    ann_i[k] = [t for t in v if t['video_id'] == v_id]
                elif k == 'videos':
                    ann_i[k] = [v_i for v_i in v if v_i['id'] == v_id]
                elif k == 'images':
                    ann_i[k] = [i for i in v if i['video_id'] == v_id]
                elif k == 'annotations':
                    ann_i[k] = [a for a in v if a['video_id'] == v_id]
                else:
                    ann_i[k] = v
            res_i = []
            for obj in res:
                if obj['video_id'] == v_id:
                    res_i.append(obj)
            tmp_ann = osp.join(res_path, 'tmp_anno.json')
            tmp_res = osp.join(res_path, 'tmp_res.json')
            json.dump(ann_i, open(tmp_ann, 'w'), indent=4)
            json.dump(res_i, open(tmp_res, 'w'), indent=4)
            teta_eval(tmp_ann, res_path, 'tmp_res.json', print_config=False)
        os.remove(tmp_ann)
        os.remove(tmp_res)

