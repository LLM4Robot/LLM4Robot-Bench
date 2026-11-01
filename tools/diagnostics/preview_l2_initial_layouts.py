#!/usr/bin/env python3
"""Render original/P01 L2 initial states in Isaac, without a policy or task run."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
TASKS = {"L2-01": "拔笔帽", "L2-02": "抽取中间书本", "L2-03": "倒玻璃球",
         "L2-04": "夹橘瓣", "L2-05": "开关抽屉", "L2-06": "夹果冻"}


def compose(output, selected):
    from PIL import Image, ImageDraw, ImageFont
    candidates = [Path("C:/Windows/Fonts/msyh.ttc"), Path("/mnt/c/Windows/Fonts/msyh.ttc"),
                  Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
                  Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")]
    font = next((p for p in candidates if p.exists()), None)
    if font is None:
        raise RuntimeError("Comparison captions need Microsoft YaHei, Noto Sans CJK, or WenQuanYi Zen Hei")
    def size(n):
        return ImageFont.truetype(str(font), n)
    cards = []
    for tid in selected:
        paths = [output / variant / tid for variant in ("before", "after")]
        if not all((p / "result.json").exists() and (p / "overview.png").exists() for p in paths):
            continue
        results = [json.loads((p / "result.json").read_text(encoding="utf-8")) for p in paths]
        passed = all(r.get("passed") for r in results)
        for view, suffix in (("overview", "comparison"), ("detail", "detail_comparison")):
            pictures = [Image.open(p / (view + ".png")).convert("RGB") for p in paths]
            w, h = pictures[0].size
            if pictures[1].size != (w, h):
                raise ValueError("Before/after resolutions differ")
            card = Image.new("RGB", (2*w+96, h+340), "#f4f7fa")
            d = ImageDraw.Draw(card)
            d.text((32,24), f"{tid} {TASKS[tid]} · P01 初始摆放对照", font=size(38), fill="#203845")
            subtitle = ("原始资产与固定相机不变 · 仅改变道具初态 · 每次复用同一组扰动"
                        if view == "overview" else
                        "道具近景自动取景，仅用于看清形态 · 相机对齐请查看全景对照图")
            d.text((32,84), subtitle, font=size(26), fill="#536e7e")
            d.text((32,136), "扰动前：原始摆放", font=size(30), fill="#3d5c6d")
            d.text((w+64,136), "扰动后：P01 初始摆放", font=size(30), fill="#087e6b")
            for x, picture in zip((32,w+64),pictures):
                card.paste(picture,(x,190))
            d.text((32,h+214), selected[tid]["description_zh"], font=size(25), fill="#294755")
            status = "两版初态检查通过" if passed else "初态检查存在问题，详见 result.json"
            d.text((32,h+268), f"{status}；未调用 GPT、未执行任务。前＝远离机械臂，后＝靠近机械臂。", font=size(23), fill="#617b8b")
            card.save(output / f"{tid}_{suffix}.png")
            if view == "overview":
                cards.append((tid, card.copy()))
    if cards:
        width=1100
        thumbs=[(t,im.resize((width,round(im.height*width/im.width)))) for t,im in cards]
        height=max(im.height for _,im in thumbs)
        sheet=Image.new("RGB",(width*2,height*((len(thumbs)+1)//2)),"#f4f7fa")
        for i,(_,im) in enumerate(thumbs):
            sheet.paste(im,((i%2)*width,(i//2)*height))
        sheet.save(output/"L2_all_comparisons.jpg",quality=95)
        sheet.save(output/"L2_all_comparisons.png")
        html='<meta charset="utf-8"><title>L2 P01 初始摆放</title><style>body{max-width:1800px;margin:24px auto;background:#f4f7fa;font-family:sans-serif}img{width:100%}a{color:#087e6b}</style>'
        for tid,_ in cards:
            html+=f'<a href="{tid}_comparison.png"><img src="{tid}_comparison.png"></a><p><a href="{tid}_detail_comparison.png">{tid} 道具近景</a></p>'
        (output/"index.html").write_text(html,encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"build/l2_p01_20261018")
    parser.add_argument("--tasks",nargs="+",choices=TASKS,default=list(TASKS))
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--compose-only",action="store_true")
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    records=json.loads((ROOT/"configs/task_references/l2_initial_p01.json").read_text(encoding="utf-8"))['tasks']
    selected={t:records[t] for t in args.tasks}
    if args.compose_only:
        compose(args.output,selected);return 0
    from robot_tasks.integrationtest.diagnostics.initial_gallery import source_fingerprint
    fingerprint=source_fingerprint()
    failures=[]
    for tid in selected:
        for variant in ("before","after"):
            folder=args.output/variant/tid;resultfile=folder/"result.json"
            old=json.loads(resultfile.read_text(encoding="utf-8")) if resultfile.exists() else {}
            if args.resume and old.get("passed") and old.get("preview",{}).get("source_sha256")==fingerprint:
                continue
            task_id = tid if variant == "before" else tid + "_v3"
            cmd=[sys.executable,"-u","-X","utf8","-m","robot_tasks.integrationtest.diagnostics.initial_gallery",task_id,"--output",str(folder)]
            print("PREVIEW_START",tid,variant,flush=True)
            with (args.output/(variant+"_"+tid+".log")).open("w",encoding="utf-8") as log:
                code=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
            result=json.loads(resultfile.read_text(encoding="utf-8")) if resultfile.exists() else {}
            passed=code==0 and result.get("passed")
            print("PREVIEW_END",tid,variant,"PASS" if passed else "FAIL",result.get("error",""),flush=True)
            if not passed:failures.append([tid,variant,code,result.get("error")])
        compose(args.output,selected)
    (args.output/"summary.json").write_text(json.dumps({"tasks":args.tasks,"failures":failures,"gpt_called":False,"remote_sync":False},indent=2),encoding="utf-8")
    return bool(failures)


if __name__=="__main__":
    raise SystemExit(main())
