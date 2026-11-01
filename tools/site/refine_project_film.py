#!/usr/bin/env python3
"""Refine the released film's presentation without retiming its recordings/audio.

Uses the mastered film from release d30871c, captions and project thumbnails. No generated
scientific imagery, optical flow or interpolation of recorded robot motion.
Requires Pillow, numpy and FFmpeg with libass. Outputs a review copy only.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
W, H, FPS = 1920, 1080, 30
INK, MUTED, SIM, REAL = '#f1f2fa', '#a3aec6', '#9c83ff', '#ff9855'


def ease(x):
    x = max(0., min(1., x))
    return x*x*(3-2*x)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ffmpeg', default='ffmpeg')
    p.add_argument('--input', type=Path, default=ROOT/'docs/media/promo/llm4robot-bench-film.mp4')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--work-dir', type=Path, required=True)
    p.add_argument('--font', default='/System/Library/Fonts/Avenir Next.ttc')
    p.add_argument('--updated-real-views', action='store_true',
                   help='Use the selected marble/whiteboard real excerpts from docs/media/loops')
    a = p.parse_args()
    expected_master='336cb59c8024bc29b4a2761ee5f4c81f1031d5581489b5a80a62941372f778d2'
    if hashlib.sha256(a.input.read_bytes()).hexdigest()!=expected_master:
        p.error('--input must be the original film from release d30871c, not an already refined copy')
    if a.input.resolve()==a.output.resolve():
        p.error('Use a separate review output; the input film must be preserved')
    a.work_dir.mkdir(parents=True, exist_ok=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    media = ROOT/'docs/media'
    fonts = {}
    replacement_frames = {}
    replacements = []
    if a.updated_real_views:
        for task, scene_start, excerpt_start in [('L2-03', 11, 1), ('L2-07', 15, 2)]:
            path = media/f'loops/{task}-real.mp4'
            # Decode only two short clips. Random access also serves the cached
            # chapter-transition frames; no synthetic motion interpolation.
            data = subprocess.check_output([a.ffmpeg, '-v', 'error', '-i', str(path),
                '-vf', f'scale=640:480,fps={FPS}', '-an', '-f', 'rawvideo',
                '-pix_fmt', 'rgb24', 'pipe:1'])
            frames = np.frombuffer(data, dtype=np.uint8).reshape((-1,480,640,3))
            # Scenes use at most six seconds of each eight-second excerpt.
            # Resampling 24 fps may omit the final fractional 30 fps frame.
            assert len(frames) >= 6*FPS, f'Short replacement excerpt: {path}'
            replacement_frames[task] = (frames, scene_start, excerpt_start)
            replacements.append({'task':task, 'domain':'real',
                'path':str(path.relative_to(ROOT)),
                'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'scene_start_seconds':scene_start, 'excerpt_start_seconds':excerpt_start,
                'source_provenance':'docs/media/loops/sources.json'})

    def real_view(task, t, fallback):
        if task not in replacement_frames:
            return fallback
        frames, scene_start, excerpt_start = replacement_frames[task]
        index = round((excerpt_start+t-scene_start)*FPS)
        assert 0 <= index < len(frames), f'Excerpt out of range: {task} at {t}'
        return Image.fromarray(frames[index])

    def font(size, bold=False):
        key = size, bold
        if key not in fonts:
            fonts[key] = ImageFont.truetype(a.font, size, index=2 if bold else 7)
        return fonts[key]

    def text(im, xy, value, size=32, color=INK, bold=False, anchor=None):
        ImageDraw.Draw(im).text(xy, value, font=font(size, bold), fill=color,
                               anchor=anchor, spacing=10)

    def panel(im, box, fill='#111827', outline='#30384b', radius=16, width=2):
        ImageDraw.Draw(im).rounded_rectangle(tuple(round(v) for v in box),
            radius, fill=fill, outline=outline, width=width)

    yy, xx = np.mgrid[:H, :W]
    glow = np.exp(-(((xx-280)/920)**2+((yy-200)/750)**2))
    pixels = np.empty((H, W, 3), dtype=np.uint8)
    for c, (base, gain) in enumerate([(9, 9), (13, 5), (23, 18)]):
        pixels[:, :, c] = base+glow*gain
    matte = Image.fromarray(pixels)
    del yy, xx, glow, pixels

    def ink_fade(old, new, u, background=matte):
        # Fade typography out before revealing the next line: no doubled text.
        if u < .5:
            return Image.blend(old, background, ease(u*2))
        return Image.blend(background, new, ease((u-.5)*2))

    def title_transition(old, new, u):
        im=Image.blend(old,new,u)
        for box in [(0,0,W,195),(0,912,W,958)]:
            im.paste(ink_fade(old.crop(box),new.crop(box),u,matte.crop(box)),box[:2])
        return im

    def base(kicker, title):
        im = matte.copy()
        text(im, (80, 50), kicker, 22, MUTED)
        text(im, (80, 100), title, 56, bold=True)
        text(im, (1840, 55), 'LLM4Robot-Bench', 22, anchor='ra')
        return im

    def tag(im, x, y, name, extra=''):
        color = SIM if name == 'SIM' else REAL
        ImageDraw.Draw(im).ellipse((x, y+9, x+10, y+19), fill=color)
        text(im, (x+22, y), name, 23, color, True)
        if extra:
            text(im, (x+110, y+1), extra, 20, MUTED)

    def paste(im, asset, box):
        x, y, w, h = (round(v) for v in box)
        im.paste(asset.resize((max(1,w), max(1,h)), Image.Resampling.LANCZOS), (x,y))

    def clean(im, t):
        im = im.copy()
        if t >= 1.2:
            # This band contains only presentation captions, never robot footage.
            # Identical captions are burned once, after all chapter transitions.
            im.paste(matte.crop((0, 958, W, H)), (0,958))
        return im

    def extract(t):
        data = subprocess.check_output([a.ffmpeg, '-v', 'error', '-ss', str(t),
            '-i', str(a.input), '-frames:v', '1', '-f', 'rawvideo',
            '-pix_fmt', 'rgb24', 'pipe:1'])
        assert len(data) == W*H*3
        return Image.frombytes('RGB', (W,H), data)

    # Cache only cut-adjacent frames, rather than a full decoded film in memory.
    cuts = [1.2,3,6,9,11,13,15,19,23,35,39,44,49,52,55]
    first = {t: extract(t) for t in cuts}
    last = {t: extract(t-1/FPS) for t in cuts}

    ids = ([f'L1-{i:02}' for i in range(1,6)] +
           [f'L2-{i:02}' for i in range(1,8)] +
           [f'L3-{i:02}' for i in range(1,5)])
    names = ['Marker Placement','Charger Alignment','Hat Hanging','Cup Nesting',
        'Flower Transfer','Marker Cap Removal','Book Pulling','Marble Pouring',
        'Orange Handling','Drawer Pull & Return','Jelly Handling','Whiteboard Wiping',
        'Bottle Cap Removal','Nut Removal','Stacked-Book Pushing & Rotation','Block Sweeping']
    thumbs = [Image.open(media/f'loops/{tid}-sim.webp').convert('RGB') for tid in ids]
    grid = base('THE TASK SUITE', '16 household tasks · 3 interaction levels')
    for i,(tid,name,thumb) in enumerate(zip(ids,names,thumbs)):
        x,y = 80+(i%4)*448, 205+(i//4)*181
        panel(grid,(x,y,x+426,y+163),radius=10)
        paste(grid,thumb,(x+2,y+2,174,130))
        text(grid,(x+191,y+20),tid,23,SIM,True)
        text(grid,(x+191,y+58),textwrap.fill(name,18),22,bold=True)
    text(grid,(80,930),'L1 · 5 tasks       L2 · 7 tasks       L3 · 4 tasks',23,MUTED)

    headers = ['Select & align contacts', 'Regulate force through contact',
               'Coordinate changing contacts']
    levels = ['Selection & alignment', 'Force regulation', 'Contact coordination']
    level_tasks = ['Marker Placement', 'Marble Pouring', 'Bottle Cap Removal']
    source_boxes = [(650,280,1220,706),(1260,280,1830,706)]
    pair_boxes = [(140,330,760,568),(1020,330,760,568)]

    def ladder(t, frame, stage=None):
        stage = stage if stage is not None else min(2,int((t-9)//2))
        stage = max(0,stage)
        im = base('THE PHYSICAL INTERACTION LADDER',headers[stage])
        since = t-(9+stage*2)
        if stage and since < .3:
            old = base('THE PHYSICAL INTERACTION LADDER',headers[stage-1])
            im = title_transition(old,im,ease(since/.3))
        for i,label in enumerate(levels):
            x=80+i*594
            active = i<=stage
            panel(im,(x,215,x+572,271),fill='#26243e' if active else '#121827',
                  outline=SIM if i==stage else '#343b50',radius=12)
            text(im,(x+20,229),f'L{i+1}',22,SIM if active else MUTED,True)
            text(im,(x+75,229),label,22,INK if active else MUTED)
        for i,(box,sbox) in enumerate(zip(pair_boxes,source_boxes)):
            tag(im,box[0],287,['SIM','REAL'][i])
            asset=frame.crop(sbox)
            if i==1 and stage==1:
                asset=real_view('L2-03',t,asset)
            paste(im,asset,box)
        text(im,(140,921),level_tasks[stage],26,bold=True)
        text(im,(1780,926),'Each level adds an interaction requirement.',21,MUTED,anchor='ra')
        return im

    def matching(t, frame):
        u=ease((t-15.22)/.55)
        im=base('ONE TASK. TWO ENVIRONMENTS.','Matched sim–real setups')
        previous=base('THE PHYSICAL INTERACTION LADDER',headers[2])
        im=title_transition(previous,im,ease((t-15.22)/.3))
        target=[(80,256,860,644),(980,256,860,644)]
        for i,(src,dst) in enumerate(zip(pair_boxes,target)):
            box=tuple(src[j]+(dst[j]-src[j])*u for j in range(4))
            tag(im,round(box[0]),round(box[1]-45),['SIM','REAL'][i])
            x,y,w,h=dst
            asset=frame.crop((x,y,x+w,y+h))
            if i==1:
                asset=real_view('L2-07',t,asset)
            paste(im,asset,box)
        text(im,(80,925),'ARX R5 · shared task and success criteria · independent timelines',21,MUTED)
        return im

    chart_base=base('THE EVALUATION','GPT-6 Astra · zero-shot')
    text(chart_base,(80,177),'3 trials per task in each environment',29,MUTED)
    tag(chart_base,1370,115,'SIM');tag(chart_base,1570,115,'REAL')
    for value in (0,20,40,60,80,100):
        y=795-value*4.5
        ImageDraw.Draw(chart_base).line((240,y,1750,y),fill='#252d40',width=1)
        text(chart_base,(215,y),f'{value}%',22,MUTED,anchor='rm')
    for i in range(3):
        x=480+i*500
        text(chart_base,(x,848),f'L{i+1}',36,bold=True,anchor='mm')
        text(chart_base,(x,895),f'{[15,21,12][i]} trials / environment',21,MUTED,anchor='mm')

    def chart(t):
        im=chart_base.copy()
        for i,(sv,rv) in enumerate([(66.7,80),(47.6,47.6),(0,0)]):
            cx=480+i*500
            # Real results follow the spoken progression; simulation follows
            # during "Simulation follows a similar trend". Labels are endpoints.
            for x,v,c,start in [(cx+25,rv,REAL,[26.6,28.2,29.8][i]),
                                (cx-125,sv,SIM,31.45+i*.22)]:
                u=ease((t-start)/.65)
                if u<=0: continue
                h=max(5,v*4.5)*u
                layer=im.copy()
                ImageDraw.Draw(layer).rounded_rectangle((x,795-h,x+100,795),
                                                        radius=min(8,h/2),fill=c)
                im=Image.blend(im,layer,min(1,u*3))
                if u>.7:
                    label=im.copy()
                    text(label,(x+50,795-max(5,v*4.5)-24),
                         f'{v:.1f}%' if v else '0%',36,c,True,anchor='mm')
                    im=Image.blend(im,label,ease((u-.7)/.3))
        if t>=32.4:
            layer=im.copy();text(layer,(1480,555),'No L3 task\ncompleted',35,bold=True,anchor='mm')
            im=Image.blend(im,layer,ease((t-32.4)/.4))
        return im

    agreement_base=base('CROSS-ENVIRONMENT RESULTS','10/16 tasks · identical success counts')
    text(agreement_base,(95,475),'10 / 16',125,bold=True)
    text(agreement_base,(100,640),'Same number of successes\nacross three trials per task.',34,MUTED)
    text(agreement_base,(100,790),'A count match—not a claim\nof identical trajectories.',24,MUTED)

    def agreement(t):
        im=agreement_base.copy()
        for i in range(16):
            x=830+(i%4)*220;y=285+(i//4)*145
            panel(im,(x,y,x+185,y+108),fill='#161d2c',outline='#30384a')
            if i<10:
                layer=im.copy()
                panel(layer,(x,y,x+185,y+108),fill='#463a74',outline=SIM)
                text(layer,(x+92,y+52),'=',50,'#d2c2ff',True,anchor='mm')
                im=Image.blend(im,layer,ease((t-35.05-i*.085)/.35))
        return im

    def scene(t, raw):
        if 6<=t<8.55:
            im=grid.copy()
            if t>8.1:
                layer=im.copy();panel(layer,(80,205,506,368),fill=None,outline=SIM,radius=10,width=4)
                im=Image.blend(im,layer,ease((t-8.1)/.4))
            return im
        if 8.55<=t<9.12:
            u=ease((t-8.55)/.57)
            dest=ladder(9,first[9],0)
            dest.paste(matte.crop((140,330,900,898)),(140,330))
            im=title_transition(grid,dest,u)
            start=(82,207,174,130);end=pair_boxes[0]
            box=tuple(start[j]+(end[j]-start[j])*u for j in range(4))
            image=Image.blend(thumbs[0].resize((760,568)),
                             first[9].crop(source_boxes[0]).resize((760,568)),u)
            paste(im,image,box)
            return im
        if 9.12<=t<15.22:
            return ladder(t,raw) if t<15 else ladder(14.999,last[15],2)
        if 15.22<=t<19:
            im=matching(t,raw)
            if t<15.42:
                im=title_transition(ladder(14.999,last[15],2),im,ease((t-15.22)/.2))
            return im
        if 23<=t<35:return chart(t)
        if 35<=t<39:return agreement(t)
        return clean(raw,t)

    # Put chapter changes in the existing pauses. Recorded clips retain their
    # native timeline. Short overlaps use existing frames, not motion synthesis.
    transitions=[(1.2,1.2,1.6),(3,3,3.42),(6,5.65,6.12),
        (19,18.77,19.18),(23,22.7,23.14),(35,34.6,35.04),
        (39,38.55,39.05),(49,48.6,49.04),(52,51.7,52.12),(55,54.65,55.1)]
    before={b:scene(b-1/FPS,last[b]) for b,_,_ in transitions}
    after={b:scene(b,first[b]) for b,_,_ in transitions}

    # Recover the existing, word-for-word captions as a final overlay so text
    # stays sharp throughout the visual transitions. Audio is stream-copied.
    srt=(media/'promo/captions.srt').read_text()
    ass=a.work_dir/'captions.ass'
    ass_header='''[Script Info]\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 1080\nWrapStyle: 0\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Default,Avenir Next,32,&H00F9F8F7,&H000000FF,&HCC090D17,&HCC090D17,0,0,0,0,100,100,0,0,3,8,0,2,180,180,27,1\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n'''
    lines=[]
    for cue in srt.strip().split('\n\n'):
        parts=cue.splitlines();start,end=parts[1].split(' --> ')
        stamp=lambda x:x.replace(',','.')[:-1]
        caption=r'\N'.join(parts[2:])
        lines.append(f'Dialogue: 0,{stamp(start)},{stamp(end)},Default,,0,0,0,,{caption}\n')
    ass.write_text(ass_header+''.join(lines))

    decoder=subprocess.Popen([a.ffmpeg,'-v','error','-i',str(a.input),'-map','0:v:0',
        '-f','rawvideo','-pix_fmt','rgb24','pipe:1'],stdout=subprocess.PIPE)
    encoder=subprocess.Popen([a.ffmpeg,'-v','error','-y','-f','rawvideo','-pix_fmt','rgb24',
        '-s',f'{W}x{H}','-r',str(FPS),'-i','pipe:0','-i',str(a.input),
        '-map','0:v:0','-map','1:a:0','-vf',f'ass={ass}','-c:v','libx264',
        '-preset','slow','-crf','19','-pix_fmt','yuv420p','-threads','4',
        '-c:a','copy','-t','60','-movflags','+faststart',str(a.output)],stdin=subprocess.PIPE)
    review_times=[0,8.4,8.7,9.1,11.2,12,15.3,15.5,16,17,18,19,27.5,29.2,32.8,35.5,38.8,44,54.9,57]
    review_frames={round(t*FPS):t for t in review_times}
    try:
        for n in range(FPS*60):
            data=decoder.stdout.read(W*H*3)
            assert len(data)==W*H*3, f'Short source at frame {n}'
            raw=Image.frombytes('RGB',(W,H),data);t=n/FPS
            im=scene(t,raw)
            for b,start,end in transitions:
                if start<=t<end:
                    outgoing=im if t<b else before[b]
                    incoming=after[b] if t<b else im
                    u=ease((t-start)/(end-start))
                    im=(ink_fade(outgoing,incoming,u) if b in (6,23,35,39,49,55)
                        else title_transition(outgoing,incoming,u))
                    break
            # Keep the recorded case change sharp; ease only the title band.
            if 44<=t<44.28:
                old=clean(last[44],44).crop((0,0,W,250))
                im.paste(ink_fade(old,im.crop((0,0,W,250)),ease((t-44)/.28),
                                  matte.crop((0,0,W,250))),(0,0))
            if n in review_frames:
                im.resize((960,540)).save(a.work_dir/f'frame-{t:05.2f}.jpg',quality=92)
            encoder.stdin.write(im.tobytes())
            if n%300==0: print(f'Rendered {n/FPS:.0f}/60 seconds',flush=True)
    finally:
        decoder.stdout.close();encoder.stdin.close()
    assert decoder.wait()==0
    assert encoder.wait()==0
    metadata={'source_master_sha256':hashlib.sha256(a.input.read_bytes()).hexdigest(),
        'output_sha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),
        'duration':60,'size':[W,H],'fps':FPS,'audio':'Original AAC stream copied unchanged',
        'captions':'Original wording and timestamps, overlaid after transitions',
        'recorded_view_replacements':replacements,
        'chapter_transitions':transitions,'review_only':True,
        'changes':['Selected task expands out of the overview grid',
            'Persistent sim/real layout across all three levels',
            'Paired view expands into the matched-environments section',
            'Real result bars follow narration; simulation appears afterwards',
            'Short eased chapter dissolves in narration pauses',
            'Original failure-clip motion and final three-second hold retained']}
    a.output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('Completed review film:',a.output,flush=True)


if __name__=='__main__':main()
