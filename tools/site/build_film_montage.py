#!/usr/bin/env python3
"""Rebuild the 16-task overview from the site's selected eight-second excerpts.

Usage: python tools/site/build_film_montage.py --ffmpeg /path/to/ffmpeg
Only presentation derivatives are touched. No experiment or source footage runs.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
MEDIA = ROOT / 'docs/media/loops'


def build(ffmpeg):
    tasks = re.findall(r"\{id:'([^']+)'", (ROOT / 'docs/films.js').read_text())
    assert len(tasks) == len(set(tasks)) == 16, 'Expected all 16 unique tasks'
    inputs = [MEDIA / f'{task}-{domain}.mp4' for task in tasks for domain in ('sim', 'real')]
    assert all(p.is_file() for p in inputs), 'Missing selected experiment excerpts'

    def run(args):
        subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', *args], check=True)

    for name, columns in (('overview', 4), ('overview-mobile', 2)):
        filters = [f'[{i}:v]scale=240:180,setsar=1,fps=24,setpts=PTS-STARTPTS[v{i}]' for i in range(32)]
        positions = [f'{(i // 2 % columns) * 480 + (i % 2) * 240}_{(i // 2 // columns) * 180}' for i in range(32)]
        filters.append(''.join(f'[v{i}]' for i in range(32)) + f'xstack=inputs=32:layout={"|".join(positions)}:shortest=1,drawgrid=w=480:h=180:t=3:c=0x080a10[out]')
        args = ['-filter_complex_threads', '2']
        for source in inputs:
            args += ['-threads', '1', '-i', str(source)]
        run([*args, '-filter_complex', ';'.join(filters), '-map', '[out]', '-an', '-t', '8',
             '-c:v', 'libx264', '-preset', 'fast', '-crf', '22', '-pix_fmt', 'yuv420p',
             '-threads', '4', '-movflags', '+faststart', str(MEDIA / f'{name}.mp4')])
        run(['-i', str(MEDIA / f'{name}.mp4'), '-frames:v', '1', '-quality', '88', str(MEDIA / f'{name}.webp')])
        print(f'{name}: {columns} × {16 // columns} task pairs', flush=True)

    run(['-i', str(MEDIA / 'overview.mp4'), '-filter_complex',
         'fps=10,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=192[p];[b][p]paletteuse=dither=bayer:bayer_scale=3',
         '-loop', '0', str(MEDIA / 'overview.gif')])
    provenance = json.loads((MEDIA / 'sources.json').read_text())
    provenance['montage_task_order'] = tasks
    (MEDIA / 'sources.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n')
    print('Downloadable GIF rebuilt', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ffmpeg', default='ffmpeg')
    build(parser.parse_args().ffmpeg)
