import torch
import numpy as np
from tqdm import tqdm
import torchvision
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class CocoDetectionEvaluator():
    def __init__(self, names, device):
        self.device = device
        # self.classes = []
        # with open(names, 'r') as f:
        #     for line in f.readlines():
        #         self.classes.append(line.strip())
        self.classes = names
    
    def coco_evaluate(self, gts, preds):
        # Create Ground Truth
        coco_gt = COCO()
        coco_gt.dataset = {}
        coco_gt.dataset["images"] = []
        coco_gt.dataset["annotations"] = []
        k = 0
        for i, gt in enumerate(gts):
            for j in range(gt.shape[0]):
                k += 1
                coco_gt.dataset["images"].append({"id": i})
                coco_gt.dataset["annotations"].append({"image_id": i, "category_id": gt[j, 0],
                                                    "bbox": np.hstack([gt[j, 1:3], gt[j, 3:5] - gt[j, 1:3]]),
                                                    "area": np.prod(gt[j, 3:5] - gt[j, 1:3]),
                                                    "id": k, "iscrowd": 0})
                
        coco_gt.dataset["categories"] = [{"id": i, "supercategory": c, "name": c} for i, c in enumerate(self.classes)]
        coco_gt.createIndex()

        # Create preadict 
        coco_pred = COCO()
        coco_pred.dataset = {}
        coco_pred.dataset["images"] = []
        coco_pred.dataset["annotations"] = []
        k = 0
        for i, pred in enumerate(preds):
            for j in range(pred.shape[0]):
                k += 1
                coco_pred.dataset["images"].append({"id": i})
                coco_pred.dataset["annotations"].append({"image_id": i, "category_id": int(pred[j, 0]),
                                                        "score": pred[j, 1], "bbox": np.hstack([pred[j, 2:4], pred[j, 4:6] - pred[j, 2:4]]),
                                                        "area": np.prod(pred[j, 4:6] - pred[j, 2:4]),
                                                        "id": k})
                
        coco_pred.dataset["categories"] = [{"id": i, "supercategory": c, "name": c} for i, c in enumerate(self.classes)]
        coco_pred.createIndex()

        coco_eval = COCOeval(coco_gt, coco_pred, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        mAP05 = coco_eval.stats[1]

        per_category_AP05 = {}
        for cat_id in coco_gt.getCatIds():
            cat_name = coco_gt.loadCats(cat_id)[0]['name']
            ap = coco_eval.eval['precision'][0, :, cat_id, 0, -1].mean()
            per_category_AP05[cat_name] = 0 if ap == -1.0 else ap

        return mAP05, per_category_AP05

    def compute_map(self, val_dataloader, model):
        gts, pts = [], []
        pbar = tqdm(val_dataloader)
        for i, (imgs, targets) in enumerate(pbar):
            # 数据预处理
            # imgs = imgs.to(self.device).float() / 255.0
            imgs = imgs.to(self.device)
            with torch.no_grad():
                # 模型预测
                preds = model(imgs)
                # 特征图后处理
                output = handle_preds(preds, self.device, 0.001)

            # 检测结果
            N, _, H, W = imgs.shape
            for p in output:
                pbboxes = []
                for b in p:
                    b = b.cpu().numpy()
                    score = b[4]
                    category = b[5]
                    x1, y1, x2, y2 = b[:4] * [W, H, W, H]
                    pbboxes.append([category, score, x1, y1, x2, y2])
                pts.append(np.array(pbboxes))

            # 标注结果
            for n in range(N):
                tbboxes = []
                for t in targets:
                    if t[0] == n:
                        t = t.cpu().numpy()
                        category = t[1]
                        bcx, bcy, bw, bh = t[2:] * [W, H, W, H]
                        x1, y1 = bcx - 0.5 * bw, bcy - 0.5 * bh
                        x2, y2 = bcx + 0.5 * bw, bcy + 0.5 * bh
                        tbboxes.append([category, x1, y1, x2, y2])
                gts.append(np.array(tbboxes))
                
        mAP05, per_category_AP05 = self.coco_evaluate(gts, pts)

        return mAP05, per_category_AP05
    
# 后处理(归一化后的坐标)
def handle_preds(preds, device, conf_thresh=0.25, nms_thresh=0.45):
    total_bboxes, output_bboxes  = [], []
    # 将特征图转换为检测框的坐标
    N, C, H, W = preds.shape
    bboxes = torch.zeros((N, H, W, 6))
    pred = preds.permute(0, 2, 3, 1)
    # 前背景分类分支
    pobj = pred[:, :, :, 0].unsqueeze(dim=-1)
    # 检测框回归分支
    preg = pred[:, :, :, 1:5]
    # 目标类别分类分支
    pcls = pred[:, :, :, 5:]

    # 检测框置信度
    bboxes[..., 4] = (pobj.squeeze(-1) ** 0.6) * (pcls.max(dim=-1)[0] ** 0.4)
    bboxes[..., 5] = pcls.argmax(dim=-1)

    # 检测框的坐标
    gy, gx = torch.meshgrid([torch.arange(H), torch.arange(W)], indexing="ij")
    bw, bh = preg[..., 2].sigmoid(), preg[..., 3].sigmoid() 
    bcx = (preg[..., 0].tanh() + gx.to(device)) / W
    bcy = (preg[..., 1].tanh() + gy.to(device)) / H

    # cx,cy,w,h = > x1,y1,x2,y1
    x1, y1 = bcx - 0.5 * bw, bcy - 0.5 * bh
    x2, y2 = bcx + 0.5 * bw, bcy + 0.5 * bh

    bboxes[..., 0], bboxes[..., 1] = x1, y1
    bboxes[..., 2], bboxes[..., 3] = x2, y2
    bboxes = bboxes.reshape(N, H*W, 6)
    total_bboxes.append(bboxes)
        
    batch_bboxes = torch.cat(total_bboxes, 1)

    # 对检测框进行NMS处理
    for p in batch_bboxes:
        output, temp = [], []
        b, s, c = [], [], []
        # 阈值筛选
        t = p[:, 4] > conf_thresh
        pb = p[t]
        for bbox in pb:
            obj_score = bbox[4]
            category = bbox[5]
            x1, y1 = bbox[0], bbox[1]
            x2, y2 = bbox[2], bbox[3]
            s.append([obj_score])
            c.append([category])
            b.append([x1, y1, x2, y2])
            temp.append([x1, y1, x2, y2, obj_score, category])
        # Torchvision NMS
        if len(b) > 0:
            b = torch.Tensor(b).to(device)
            c = torch.Tensor(c).squeeze(1).to(device)
            s = torch.Tensor(s).squeeze(1).to(device)
            keep = torchvision.ops.batched_nms(b, s, c, nms_thresh)
            for i in keep:
                output.append(temp[i])
        output_bboxes.append(torch.Tensor(output))
    return output_bboxes