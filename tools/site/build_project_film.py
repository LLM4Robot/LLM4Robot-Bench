#!/usr/bin/env python3
"""Render the 60-second project film from recorded footage and timed narration.

Presentation only: original recordings are read, never modified. Supply the
source clips named in docs/media/promo/sources.json and the narration segments
listed in narration.json. Requires Pillow, numpy and FFmpeg with libass.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import textwrap

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[2]
W, H, FPS = 1920, 1080, 30
BG = '#090d17'
INK = '#f1f2fa'
MUTED = '#a3aec6'
SIM = '#9c83ff'
REAL = '#ff9855'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ffmpeg', default='ffmpeg')
    p.add_argument('--sources', type=Path, required=True)
    p.add_argument('--narration', type=Path, required=True)
    p.add_argument('--work-dir', type=Path, required=True)
    p.add_argument('--font', default='/System/Library/Fonts/Avenir Next.ttc')
    p.add_argument('--only-stills', action='store_true')
    p.add_argument('--reuse-scenes', action='store_true', help='Reuse existing picture renders when only narration/captions changed')
    a = p.parse_args()
    work = a.work_dir; work.mkdir(parents=True, exist_ok=True)
    out = ROOT / 'docs/media/promo'; out.mkdir(parents=True, exist_ok=True)
    media = ROOT / 'docs/media'
    fonts = {}
    def font(size, bold=False):
        key=(size,bold)
        if key not in fonts: fonts[key]=ImageFont.truetype(a.font,size,index=2 if bold else 7)
        return fonts[key]
    def text(im, pos, value, size=32, color=INK, bold=False, anchor=None):
        ImageDraw.Draw(im).text(pos,value,font=font(size,bold),fill=color,anchor=anchor,spacing=10)
    def line(im,xy,fill='#31384c',width=2): ImageDraw.Draw(im).line(xy,fill=fill,width=width)
    def panel(im,box,fill='#111827',outline='#30384b',radius=20):
        ImageDraw.Draw(im).rounded_rectangle(box,radius,fill=fill,outline=outline,width=2)
    def background():
        im=Image.new('RGB',(W,H),BG)
        # A very soft violet-to-ink matte keeps the frame close to the website.
        import numpy as np
        yy,xx=np.mgrid[:H,:W];glow=np.exp(-(((xx-280)/920)**2+((yy-200)/750)**2))
        ar=np.zeros((H,W,3),dtype=np.uint8)
        for c,(base,gain) in enumerate([(9,9),(13,5),(23,18)]): ar[:,:,c]=base+glow*gain
        return Image.fromarray(ar)
    matte=background()
    def base(kicker,title):
        im=matte.copy();text(im,(80,50),kicker,22,MUTED);text(im,(80,100),title,56,bold=True)
        text(im,(1840,55),'LLM4Robot-Bench',22,INK,anchor='ra')
        return im
    def fit(im,path,box):
        asset=Image.open(path).convert('RGB');x,y,w,h=box
        asset=ImageOps.contain(asset,(w,h),Image.Resampling.LANCZOS)
        im.paste(asset,(x+(w-asset.width)//2,y+(h-asset.height)//2))
    def tag(im,x,y,domain,extra=''):
        c=SIM if domain=='SIM' else REAL
        ImageDraw.Draw(im).ellipse((x,y+9,x+10,y+19),fill=c)
        text(im,(x+22,y),domain,23,c,True)
        if extra:text(im,(x+110,y+1),extra,20,MUTED)
    def twin(kicker,title,note='',extras=('','')):
        im=base(kicker,title)
        for x,d,e in [(80,'SIM',extras[0]),(980,'REAL',extras[1])]:
            tag(im,x,210,d,e);panel(im,(x-1,255,x+861,900),outline=SIM if d=='SIM' else REAL,radius=4)
        if note:text(im,(80,924),note,21,MUTED)
        return im
    def ladder(im,level=3,small=False):
        x=85;bottom=830;width=430 if not small else 340
        labels=[('L1','Select & align contacts'),('L2','+ Force regulation'),('L3','+ Coordinated contact changes')]
        for i,(number,label) in enumerate(labels[:level]):
            y=bottom-((i+1)*165)
            panel(im,(x+i*22,y,x+width+i*22,y+136),fill=['#172e49','#212b53','#302750'][i],outline=['#5287be','#7186e0','#ab91ff'][i],radius=13)
            text(im,(x+22+i*22,y+15),number,30,SIM,True)
            wrapped=textwrap.fill(label,23 if not small else 19)
            text(im,(x+22+i*22,y+61),wrapped,23,INK,True)
    def film(path,start,box,crop=None,length=None):
        return {'path':str(path),'start':start,'box':box,'crop':crop,'length':length}
    sources=[];scenes=[]
    def add(name,start,end,im,clips=(),overlay=None):
        fn=work/(name+'.png');im.save(fn)
        ov=None
        if overlay is not None:ov=work/(name+'-overlay.png');overlay.save(ov)
        scenes.append({'name':name,'start':start,'end':end,'image':fn,'clips':list(clips),'overlay':ov})
    # First frame and poster retain the complete submitted Figure 1.
    opening=matte.copy();fit(opening,media/'paper/figure1.png',(48,12,1824,1056))
    opening.save(out/'poster.webp',quality=94)
    add('00-figure1',0,1.2,opening)
    cap_s=a.sources/'cap-sim.mp4';cap_r=a.sources/'cap-real.mp4'
    sw_s=a.sources/'sweep-sim.mp4';sw_r=a.sources/'sweep-real.mp4'
    capcrop=(248,480,704,528)
    standard=[(80,256,860,644),(980,256,860,644)]
    add('01-question',1.2,3,twin('A FRONTIER MODEL. A PHYSICAL WORLD.','How high can Astra climb?',extras=('1× simulated time','10× execution')),
        [film(cap_s,88,standard[0],capcrop),film(cap_r,66,standard[1],capcrop)])
    intro=Image.new('RGBA',(W,H),(4,7,17,177));text(intro,(W//2,415),'LLM4Robot-Bench',120,INK,True,anchor='mm');text(intro,(W//2,520),'A matched sim–real benchmark',43,'#d4ccec',anchor='mm')
    add('02-introduction',3,6,matte.copy(),[film(cap_s,89.8,standard[0],capcrop),film(cap_r,67.8,standard[1],capcrop)],intro)
    taskids=[f'L1-{n:02}' for n in range(1,6)]+[f'L2-{n:02}' for n in range(1,8)]+[f'L3-{n:02}' for n in range(1,5)]
    titles=['Marker Placement','Charger Alignment','Hat Hanging','Cup Nesting','Flower Transfer','Marker Cap Removal','Book Pulling','Marble Pouring','Orange Handling','Drawer Pull & Return','Jelly Handling','Whiteboard Wiping','Bottle Cap Removal','Nut Removal','Stacked-Book Pushing & Rotation','Block Sweeping']
    grid=base('THE TASK SUITE','16 household tasks · 3 interaction levels')
    # Group sizes 5 / 7 / 4 are explicit, while all sixteen frames remain legible.
    for i,(tid,title) in enumerate(zip(taskids,titles)):
        col=i%4;row=i//4;x=80+col*448;y=205+row*181
        panel(grid,(x,y,x+426,y+163),radius=10)
        fit(grid,media/f'loops/{tid}-sim.webp',(x+2,y+2,174,130))
        text(grid,(x+191,y+20),tid,23,SIM,True)
        text(grid,(x+191,y+58),textwrap.fill(title,18),22,INK,True)
    text(grid,(80,956),'L1 · 5 tasks       L2 · 7 tasks       L3 · 4 tasks',23,MUTED)
    add('03-task-suite',6,9,grid)
    for n,tid,start,end,title in [(1,'L1-01',9,11,'Select & align contacts'),(2,'L2-03',11,13,'Regulate force through contact'),(3,'L3-01',13,15,'Coordinate changing contacts')]:
        im=base('THE PHYSICAL INTERACTION LADDER',title);ladder(im,n)
        x,y=650,280;tag(im,x,y-44,'SIM');tag(im,x+610,y-44,'REAL')
        clips=[film(media/f'loops/{tid}-{d}.mp4',1,(x+j*610,y,570,427)) for j,d in enumerate(('sim','real'))]
        if n==3:
            clips=[film(cap_s,89,(x,y,570,427),(620,116,560,316)),film(cap_r,66,(x+610,y,570,427),(20,116,560,316))]
        text(im,(650,755),titles[taskids.index(tid)],33,INK,True)
        text(im,(650,815),'Each level adds an interaction requirement.',25,MUTED)
        add(f'0{n+3}-ladder-{n}',start,end,im,clips)
    im=twin('ONE TASK. TWO ENVIRONMENTS.','Matched sim–real setups','ARX R5 · shared task and success criteria · independent timelines')
    add('07-matching',15,19,im,[film(media/f'loops/L2-07-{d}.mp4',2,standard[i]) for i,d in enumerate(('sim','real'))])
    im=base('PHYSICAL PROPERTIES','Mass set from measurements')
    for i,d in enumerate(('sim','real')):
        x=80+900*i;tag(im,x,205,d);panel(im,(x,255,x+860,790),fill='#f5f6f9',radius=10)
        fit(im,media/f'alignment/mass-{d}.webp',(x+24,272,812,490));text(im,(x+430,852),'26.4 g',66, SIM if i==0 else REAL,True,anchor='mm')
    text(im,(960,925),'Measured mass  /  configured mass',25,MUTED,anchor='mm')
    add('08-mass',19,23,im)
    def chart(progress=0):
        im=base('THE EVALUATION','GPT-6 Astra · zero-shot')
        text(im,(80,177),'3 trials per task in each environment',29,MUTED)
        tag(im,1370,115,'SIM');tag(im,1570,115,'REAL')
        left,bottom=240,795
        for value in (0,20,40,60,80,100):
            y=bottom-value*4.5;line(im,(left,y,1750,y),'#252d40',1);text(im,(left-25,y),str(value)+'%',22,MUTED,anchor='rm')
        for i,(sv,rv) in enumerate([(66.7,80),(47.6,47.6),(0,0)]):
            cx=480+i*500
            text(im,(cx,bottom+53),f'L{i+1}',36,INK,True,anchor='mm')
            text(im,(cx,bottom+100),f'{[15,21,12][i]} trials / environment',21,MUTED,anchor='mm')
            if progress<i+1: continue
            for x,v,c in [(cx-125,sv,SIM),(cx+25,rv,REAL)]:
                h=max(5,v*4.5);ImageDraw.Draw(im).rounded_rectangle((x,bottom-h,x+100,bottom),8,fill=c)
                text(im,(x+50,bottom-h-24),f'{v:.1f}%' if v else '0%',36,c,True,anchor='mm')
            if i==2:text(im,(cx,555),'No L3 task\ncompleted',35,INK,True,anchor='mm')
        return im
    add('09-protocol',23,26,chart())
    add('10-results-l1',26,27,chart(1));add('11-results-l2',27,28,chart(2));add('12-results-l3',28,35,chart(3))
    im=base('CROSS-ENVIRONMENT RESULTS','10/16 tasks · identical success counts')
    # No task identity is inferred from an aggregate count.
    for i in range(16):
        x=830+(i%4)*220;y=285+(i//4)*145
        panel(im,(x,y,x+185,y+108),fill='#463a74' if i<10 else '#161d2c',outline=SIM if i<10 else '#30384a')
        if i<10:text(im,(x+92,y+52),'=',50,'#d2c2ff',True,anchor='mm')
    text(im,(95,475),'10 / 16',125,INK,True);text(im,(100,640),'Same number of successes\nacross three trials per task.',34,MUTED)
    text(im,(100,790),'A count match—not a claim\nof identical trajectories.',24,MUTED)
    add('13-agreement',35,39,im)
    im=twin('ILLUSTRATIVE ROLLOUTS · L3-01','Local progress. Then contact slips.','Bottle Cap Removal · independent timelines',('1× simulated time','10× execution'))
    clips=[film(cap_s,88,standard[0],capcrop,4),film(cap_r,66,standard[1],capcrop,4),film(cap_s,88,(626,265,300,169),(620,116,560,316),4),film(cap_r,66,(1526,265,300,169),(20,116,560,316),4)]
    add('14-cap-failure',39,44,im,clips)
    im=twin('ILLUSTRATIVE ROLLOUTS · L3-04','Coordination falters.','Block Sweeping · source playback retained · execution speed not recorded',('recorded excerpt','recorded excerpt'))
    clips=[film(sw_s,392,standard[0],(0,422,1080,804),4),film(sw_r,49,standard[1],(208,310,544,410),4),film(sw_s,392,(626,265,300,169),(0,84,540,304),4),film(sw_r,49,(1526,265,300,169),(0,18,480,270),4)]
    add('15-sweep-failure',44,49,im,clips)
    im=base('WHAT THESE RESULTS SUPPORT','For Astra: similar trends and limitations')
    ladder(im,3)
    text(im,(830,340),'L1  /  L2  /  L3',55,INK,True)
    tag(im,850,470,'SIM');text(im,(1010,470),'66.7%  /  47.6%  /  0%',40,SIM,True)
    tag(im,850,555,'REAL');text(im,(1010,555),'80.0%  /  47.6%  /  0%',40,REAL,True)
    text(im,(850,718),'No L3 task completed\nin either environment.',38,INK,True)
    add('16-takeaway',49,52,im)
    im=base('WHAT THESE RESULTS SUPPORT','From physical interaction to a repeatable testbed')
    ladder(im,3,True)
    text(im,(790,340),'Matched simulation',66,INK,True)
    text(im,(790,438),'as a scalable testbed.',60,SIM,True)
    text(im,(790,585),'Supported for Astra\nin this evaluation.',34,MUTED)
    add('17-testbed',52,55,im)
    im=matte.copy();text(im,(960,175),'LLM4Robot-Bench',125,INK,True,anchor='mm');text(im,(960,315),'How high can your model climb?',50,'#c9c2e2',anchor='mm')
    panel(im,(460,422,1460,515),fill='#171d2b',outline='#695d94',radius=47)
    text(im,(960,467),'llm4robot.github.io/bench/',45,INK,True,anchor='mm')
    # Transparent logo sheet is rendered directly from the site's original-color
    # images and existing SVG masks, not recolored or reconstructed.
    if (a.sources/'affiliations.png').exists():
        logos=Image.open(a.sources/'affiliations.png').convert('RGBA');logos.thumbnail((1720,300),Image.Resampling.LANCZOS)
        im.paste(logos,((W-logos.width)//2,610),logos)
    text(im,(960,925),'Author affiliations',19,MUTED,anchor='mm')
    add('18-end-card',55,60,im)
    # A contact sheet supports a visual review without reading every frame.
    contact=Image.new('RGB',(1280,math.ceil(len(scenes)/4)*200),BG)
    for i,s in enumerate(scenes):
        thumb=Image.open(s['image']).resize((320,180));contact.paste(thumb,(i%4*320,i//4*200));text(contact,(i%4*320+5,i//4*200+181),s['name'],12)
    contact.save(work/'storyboard.jpg',quality=90)
    if a.only_stills:return
    def run(args):subprocess.run([a.ffmpeg,'-hide_banner','-loglevel','error','-y',*map(str,args)],check=True)
    scene_files=[]
    for s in scenes:
        duration=s['end']-s['start'];dest=work/(s['name']+'.mp4');scene_files.append(dest)
        if a.reuse_scenes and dest.exists(): continue
        args=['-loop','1','-framerate',FPS,'-i',s['image']];filts=[];last='0:v';idx=1
        for c in s['clips']:
            args+=['-ss',c['start'],'-t',c['length'] or duration,'-i',c['path']]
            x,y,w,h=c['box'];w=w//2*2;h=h//2*2;crop=c['crop'];f=f'[{idx}:v]'
            if crop:
                cx,cy,cw,ch=crop;f+=f'crop={cw}:{ch}:{cx}:{cy},'
            f+=f'scale={w}:{h}:force_original_aspect_ratio=decrease:force_divisible_by=2,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=0x090d17,setsar=1,fps={FPS},setpts=PTS-STARTPTS'
            if c['length']:f+=f',tpad=stop_mode=clone:stop_duration={duration-c["length"]+.1}'
            f+=f'[v{idx}]';filts.append(f);filts.append(f'[{last}][v{idx}]overlay={x}:{y}:eof_action=repeat[o{idx}]');last=f'o{idx}';idx+=1
        if s['overlay']:
            args+=['-loop','1','-framerate',FPS,'-i',s['overlay']]
            filts.append(f'[{last}][{idx}:v]overlay=0:0[out]');last='out'
        if filts:args+=['-filter_complex_threads','2','-filter_complex',';'.join(filts),'-map',f'[{last}]']
        args+=['-t',duration,'-an','-c:v','libx264','-crf','18','-preset','fast','-pix_fmt','yuv420p','-threads','4','-r',FPS,dest]
        run(args);print('Rendered',s['name'],flush=True)
    concat=work/'concat.txt';concat.write_text(''.join(f"file '{v}'\n" for v in scene_files))
    silent=work/'picture.mp4';run(['-f','concat','-safe','0','-i',concat,'-c','copy',silent])
    rows=json.loads(a.narration.read_text());args=[];filters=[]
    for i,row in enumerate(rows):
        args+=['-i',a.narration.parent/row['file']]
        # TTS files include trailing silence; trim it, never accelerate speech.
        end=(row['words'][-1]['offset']+row['words'][-1]['duration'])/1e7+.07
        filters.append(f'[{i}:a]atrim=0:{end},asetpts=PTS-STARTPTS,adelay={round(row["start"]*1000)}:all=1[a{i}]')
    filters.append(''.join(f'[a{i}]' for i in range(len(rows)))+f'amix=inputs={len(rows)}:normalize=0,apad,atrim=0:60,loudnorm=I=-17:TP=-1.5:LRA=8[aout]')
    run([*args,'-filter_complex',';'.join(filters),'-map','[aout]','-ar','48000',work/'narration.wav'])
    # Phrase captions carry the narrated story even when autoplay is muted.
    caps=[]
    for row_index,row in enumerate(rows):
        words=row['words']
        if row_index == 11:
            parts=[(words[:5], 'These illustrative rollouts show how'),
                   (words[5:], 'local progress can break down\nas contact slips')]
        else:
            parts=[(words,row['text'])]
        for selected, phrase in parts:
            begin=max(1.2,row['start']+selected[0]['offset']/1e7)
            end=row['start']+(selected[-1]['offset']+selected[-1]['duration'])/1e7+.05
            wrapped=phrase if '\n' in phrase else textwrap.fill(phrase,42)
            assert len(wrapped.splitlines())<=2, phrase
            caps.append((begin,end,wrapped))
    caps=[(b,min(e,caps[i+1][0]-.02) if i+1<len(caps) else e,t)
          for i,(b,e,t) in enumerate(caps)]
    # Hold the final invitation so the last three seconds are fully static.
    caps[-1] = (caps[-1][0], 60, caps[-1][2])
    def ts(t,comma=False):
        ms=round(t*1000);h,ms=divmod(ms,3600000);m,ms=divmod(ms,60000);s,ms=divmod(ms,1000)
        return f'{h:02}:{m:02}:{s:02}{"," if comma else "."}{ms:03}'
    (out/'captions.vtt').write_text('WEBVTT\n\n'+''.join(f'{ts(b)} --> {ts(e)}\n{t}\n\n' for b,e,t in caps))
    (out/'captions.srt').write_text(''.join(f'{i+1}\n{ts(b,True)} --> {ts(e,True)}\n{t}\n\n' for i,(b,e,t) in enumerate(caps)))
    ass=work/'captions.ass'
    header='''[Script Info]\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 1080\nWrapStyle: 0\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Default,Avenir Next,32,&H00F9F8F7,&H000000FF,&HCC090D17,&HCC090D17,0,0,0,0,100,100,0,0,3,8,0,2,180,180,27,1\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n'''
    def ass_time(t):return ts(t)[:-1]
    ass.write_text(header+''.join(f'Dialogue: 0,{ass_time(b)},{ass_time(e)},Default,,0,0,0,,{t.replace(chr(10),chr(92)+"N")}\n' for b,e,t in caps))
    run(['-i',silent,'-i',work/'narration.wav','-vf',f'ass={ass}','-c:v','libx264','-preset','slow','-crf','20','-threads','4','-pix_fmt','yuv420p','-c:a','aac','-b:a','160k','-t','60','-movflags','+faststart',out/'llm4robot-bench-film.mp4'])
    run(['-i',out/'llm4robot-bench-film.mp4','-vf','scale=1280:720','-an','-c:v','libx264','-preset','slow','-crf','25','-threads','4','-movflags','+faststart',out/'llm4robot-bench-preview.mp4'])
    transcript='\n\n'.join(row['text'] for row in rows)
    (out/'transcript.txt').write_text(transcript+'\n')
    provenance={'title':'LLM4Robot-Bench — 60-second project film','duration_seconds':60,'size':[1920,1080],'fps':30,'design_document':'https://sxavxhkemv.feishu.cn/docx/N3drdFzrIor77axFaWbcmfpfngf','opening':'Complete submitted Figure 1; no content retouched.','narration':'Synthetic English narration, en-US-AriaNeural, original supplied script. Trailing silence trimmed; speech not time-compressed.','footage':'Recorded project rollouts; illustrative excerpts do not identify the scored Table 2 episodes. Simulation and real footage have independent timelines. Existing excerpts retain source playback speed.','source_files':[],'scenes':[]}
    source_names = {
        'cap-sim.mp4': 'gpt_run_L3-01_20261018_three_views',
        'cap-real.mp4': 'left_cap_unscrewing_trial02_full_3view_90s',
        'sweep-sim.mp4': 'L3-05_第二次_三视角合成',
        'sweep-real.mp4': 'L3-04_扫积木入簸箕_失败_independent_brush_20261018_183232',
    }
    provenance['materials_document'] = 'https://my.feishu.cn/docx/U8AsdPiq2oqRb8xzletcrsXanBh'
    provenance['existing_assets'] = ['docs/media/loops/sources.json', 'docs/media/alignment/sources.json', 'docs/media/paper/source.json']
    for filename, original in source_names.items():
        src = a.sources / filename
        provenance['source_files'].append({'file':filename,'original_name':original,'sha256':hashlib.sha256(src.read_bytes()).hexdigest()})
    for s in scenes:
        provenance['scenes'].append({'name':s['name'],'start':s['start'],'end':s['end'],'clips':[{**c,'path':Path(c['path']).name} for c in s['clips']]})
    for name in ['llm4robot-bench-film.mp4','llm4robot-bench-preview.mp4','poster.webp']:
        provenance[name]={'bytes':(out/name).stat().st_size,'sha256':hashlib.sha256((out/name).read_bytes()).hexdigest()}
    (out/'sources.json').write_text(json.dumps(provenance,indent=2,ensure_ascii=False)+'\n')
    print('Completed project film:',out,flush=True)

if __name__=='__main__':main()
