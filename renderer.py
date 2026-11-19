"""ChoiceForge Studio — offline, deterministic original video renderer.

Python + Pillow/OpenCV render RGB frames; FFmpeg encodes and optionally mixes licensed music.
No copyrighted tracks or third-party backgrounds are bundled.
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from narration import render_voice, voice_settings

ROOT = Path(__file__).resolve().parent
FONTS = [
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    'C:/Windows/Fonts/arial.ttf',
    '/System/Library/Fonts/Supplemental/Arial.ttf',
    '/usr/share/fonts/opentype/inter/Inter-Regular.otf',
]
FONT_BOLD = [
    '/usr/share/fonts/opentype/inter/InterDisplay-Bold.otf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
    'C:/Windows/Fonts/arialbd.ttf',
    '/System/Library/Fonts/Supplemental/Arial Bold.ttf',
]
FONT_HEAVY = [
    '/usr/share/fonts/opentype/inter/InterDisplay-Black.otf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
    'C:/Windows/Fonts/arialbd.ttf',
    '/System/Library/Fonts/Supplemental/Arial Bold.ttf',
]

THEMES = {
    'Electric Midnight': {'top':(15,18,46), 'bottom':(37,10,66), 'left':(42,210,251), 'right':(255,86,173), 'label':'ELECTRIC / INTERACTIVE', 'mood':['energetic','upbeat','electronic']},
    'Candy Pop': {'top':(58,20,100), 'bottom':(15,36,95), 'left':(255,186,51), 'right':(255,86,151), 'label':'CANDY / POP CULTURE', 'mood':['upbeat','playful','pop']},
    'Ocean vs Sunset': {'top':(12,37,65), 'bottom':(30,17,47), 'left':(28,208,194), 'right':(252,127,87), 'label':'OCEAN / SUNSET', 'mood':['chill','electronic','upbeat']},
    'Studio Black': {'top':(12,14,20), 'bottom':(26,31,39), 'left':(192,233,255), 'right':(255,205,104), 'label':'MINIMAL / PREMIUM', 'mood':['minimal','ambient','cinematic']},
}

DEFAULT_QUESTIONS = [
    {'a':'Be able to pause time for 10 minutes a day', 'b':'Teleport to any place once a week', 'note':'One changes your time. The other changes your world.'},
    {'a':'Live in a beautiful house by the ocean', 'b':'Live in a sky-high apartment in Tokyo', 'note':'Peace or nonstop excitement?'},
    {'a':'Know every language on Earth', 'b':'Play every musical instrument perfectly', 'note':'Which talent would transform your life?'},
    {'a':'Explore the deepest ocean', 'b':'Walk on the surface of Mars', 'note':'The unknown is closer than you think.'},
    {'a':'Have unlimited free flights', 'b':'Have unlimited free concerts', 'note':'Adventure or unforgettable nights?'},
]


def font(size: int, bold=False, heavy=False):
    choices = FONT_HEAVY if heavy else FONT_BOLD if bold else FONTS
    for item in choices:
        if Path(item).exists():
            return ImageFont.truetype(item, size)
    return ImageFont.load_default()


def validate_config(cfg: dict):
    required = {'title', 'questions', 'theme', 'aspect', 'seconds_per_question'}
    if not required.issubset(cfg):
        raise ValueError(f'Missing project fields: {sorted(required - set(cfg))}')
    if cfg['aspect'] not in ('9:16', '16:9'):
        raise ValueError('Aspect must be 9:16 or 16:9')
    if cfg['theme'] not in THEMES:
        raise ValueError(f"Theme must be one of: {', '.join(THEMES)}")
    if not 2 <= float(cfg['seconds_per_question']) <= 20:
        raise ValueError('seconds_per_question must be between 2 and 20')
    voice_settings(cfg)
    if cfg.get('motion','Gentle') not in ('Off','Gentle','Dynamic'):
        raise ValueError('Motion must be Off, Gentle or Dynamic')
    if not 1 <= len(cfg['questions']) <= 25:
        raise ValueError('Use 1–25 original questions per video')
    for idx, q in enumerate(cfg['questions'], 1):
        if not isinstance(q, dict) or not q.get('a') or not q.get('b'):
            raise ValueError(f'Question {idx} must contain a and b')
        for key in ('a', 'b'):
            if len(str(q[key])) > 200:
                raise ValueError(f'Question {idx}, option {key} is too long (200-character max)')


def canvas_dims(aspect: str, quality: str):
    if aspect == '9:16':
        return (1080,1920) if quality == 'Full HD' else (540,960)
    return (1920,1080) if quality == 'Full HD' else (960,540)


def blend_rgba(im: Image.Image, layer: Image.Image, offset=(0,0)):
    im.alpha_composite(layer, offset)


def bg_image(w:int,h:int,theme:str, photo: Optional[str]=None, seed:int=17):
    p=THEMES[theme]
    t=np.linspace(0,1,h,dtype=np.float32)[:,None,None]
    aa=np.asarray(p['top'], dtype=np.float32)[None,None,:]
    bb=np.asarray(p['bottom'],dtype=np.float32)[None,None,:]
    base=np.repeat((aa*(1-t)+bb*t).astype(np.uint8),w,axis=1)
    img=Image.fromarray(base,'RGB').convert('RGBA')
    gl=Image.new('RGBA',(w,h),(0,0,0,0)); d=ImageDraw.Draw(gl,'RGBA')
    size=min(w,h)
    d.ellipse((-int(.65*w),int(.07*h),int(.68*w),int(.68*h)), fill=(*p['left'],105))
    d.ellipse((int(.48*w),int(.3*h),int(1.2*w),int(.91*h)),fill=(*p['right'],90))
    gl=gl.filter(ImageFilter.GaussianBlur(radius=int(size*.14)))
    img.alpha_composite(gl)
    if photo and Path(photo).is_file():
        with Image.open(photo) as raw:
            raw=raw.convert('RGB')
            src_ratio=raw.width/raw.height; target=w/h
            if src_ratio > target:
                nh=h; nw=round(h*src_ratio)
            else:
                nw=w; nh=round(w/src_ratio)
            fitted=raw.resize((nw,nh),Image.Resampling.LANCZOS)
            fitted=fitted.crop(((nw-w)//2,(nh-h)//2,(nw+w)//2,(nh+h)//2))
            img=Image.blend(img, fitted.convert('RGBA'), .30)
    # Elegant faint grid. No stock image needed.
    grid=Image.new('RGBA',(w,h),(0,0,0,0))
    draw=ImageDraw.Draw(grid,'RGBA')
    step=max(36,int(w*.1))
    for x in range(0,w,step):draw.line((x,0,x,h),fill=(255,255,255,15),width=1)
    for y in range(0,h,step):draw.line((0,y,w,y),fill=(255,255,255,15),width=1)
    rng=random.Random(seed)
    for _ in range(25):
        x=rng.randint(0,w);y=rng.randint(0,h)
        rr=rng.randint(1,3)
        draw.ellipse((x-rr,y-rr,x+rr,y+rr),fill=(255,255,255,50))
    img.alpha_composite(grid)
    return img


def fit_lines(text, max_width, size, min_size=20, max_lines=5):
    text=str(text).strip()
    f=font(size,bold=True)
    scratch=Image.new('L',(1,1));d=ImageDraw.Draw(scratch)
    while size>=min_size:
        f=font(size,bold=True)
        words=text.split()
        out=[]; line=''
        for word in words:
            candidate=(line+' '+word).strip()
            if d.textbbox((0,0),candidate,font=f)[2] <= max_width:
                line=candidate
            else:
                if line: out.append(line)
                line=word
        if line:out.append(line)
        if len(out)<=max_lines and all(d.textbbox((0,0),l,font=f)[2]<=max_width for l in out):
            return out,f
        size-=2
    raise ValueError('Text cannot fit in option panel; shorten the question or use landscape output')


def pill(draw,xy,text,font_,color,fill=None,radius=20,stroke=None):
    l,t,r,b=xy
    draw.rounded_rectangle((l,t,r,b),radius=radius,fill=fill,outline=stroke,width=2)
    bb=draw.textbbox((0,0),text,font=font_)
    tw=bb[2]-bb[0];th=bb[3]-bb[1]
    draw.text(((l+r-tw)/2,(t+b-th)/2-bb[1]),text,font=font_,fill=color)


def draw_centered(draw,xy,text,font_,color,spacing=8):
    lines=text if isinstance(text,list) else [text]
    fbb=draw.textbbox((0,0),'Ay',font=font_)
    height=(fbb[3]-fbb[1])+spacing
    x,y=xy
    for j,line in enumerate(lines):
        bb=draw.textbbox((0,0),line,font=font_)
        draw.text((int(x-(bb[2]-bb[0])/2),int(y+j*height)),line,font=font_,fill=color)


def opt_panel(size, color, label, text, image_path=None, horizontal=False):
    """Individual editable panel, to be placed in the final composition."""
    w,h=size; out=Image.new('RGBA',(w,h),(0,0,0,0));d=ImageDraw.Draw(out,'RGBA')
    radius=max(18,int(min(w,h)*.065))
    d.rounded_rectangle((2,2,w-3,h-3),radius=radius,fill=(255,255,255,25),outline=(*color,245),width=max(3,int(w*.007)))
    d.rounded_rectangle((10,10,w-10,h-10),radius=radius,fill=(*color,23))
    if image_path and Path(image_path).is_file():
        with Image.open(image_path) as source:
            source=source.convert('RGBA')
            box=(max(25,int(w*.06)),max(24,int(h*.18)),w-max(25,int(w*.06)),int(h*.60))
            iw,ih=box[2]-box[0],box[3]-box[1]
            srcw,srch=source.size; ratio=max(iw/srcw,ih/srch)
            fitted=source.resize((int(srcw*ratio)+1,int(srch*ratio)+1),Image.Resampling.LANCZOS)
            xx=(fitted.width-iw)//2; yy=(fitted.height-ih)//2
            fitted=fitted.crop((xx,yy,xx+iw,yy+ih))
            mask=Image.new('L',(iw,ih),0);md=ImageDraw.Draw(mask)
            md.rounded_rectangle((0,0,iw,ih),radius=max(12,int(w*.03)),fill=255)
            out.paste(fitted,(box[0],box[1]),mask)
    dot=max(28,int(h*.105))
    pill(d,(int(w*.08),int(h*.07),int(w*.08)+dot,int(h*.07)+dot),label,font(int(dot*.48),heavy=True),(15,17,35),fill=(*color,255),radius=dot//2)
    maxw=int(w*.83)
    if image_path and Path(image_path).is_file():
        availh=h*.26; cy=h*.73; default=int(min(w*.084,h*.16))
    else:
        availh=h*.55; cy=h*.52; default=int(min(w*.084,h*.16))
    lines,f=fit_lines(text,maxw,default,min_size=max(12,int(w*.028)),max_lines=5)
    line_h=int(f.size*1.22)
    y0=cy-len(lines)*line_h/2
    for j,line in enumerate(lines):
        bb=d.textbbox((0,0),line,font=f)
        tw=bb[2]-bb[0]
        d.text(((w-tw)/2,y0+j*line_h),line,font=f,fill=(255,255,255,255))
    # Bottom gradient-like bar for visual identity
    d.rounded_rectangle((int(w*.18),h-int(h*.075),int(w*.82),h-int(h*.06)),radius=8,fill=(*color,170))
    return out


def round_scene(cfg,w,h,index):
    p=THEMES[cfg['theme']];q=cfg['questions'][index]
    base=bg_image(w,h,cfg['theme'],cfg.get('background_image'),seed=17+index)
    typography=Image.new('RGBA',(w,h),(0,0,0,0))
    d=ImageDraw.Draw(typography,'RGBA'); short=min(w,h)
    portrait=w<h
    margin=int(w*(.065 if portrait else .07))
    rbadge_h=int(short*.065)
    pill(d,(margin,int(h*.058),margin+int(w*.33),int(h*.058)+rbadge_h),f'ROUND  {index+1:02d} / {len(cfg["questions"]):02d}',font(int(short*.026),bold=True),(230,240,255,255),fill=(255,255,255,27),stroke=(255,255,255,70),radius=18)
    eyebrow='WOULD YOU RATHER?' if cfg.get('format','Would You Rather')=='Would You Rather' else cfg.get('format','THIS OR THAT?').upper()
    if cfg.get('format')=='Pick One':eyebrow='PICK ONE. NO EXPLANATIONS.'
    head=font(int(short*(.081 if portrait else .055)),heavy=True)
    draw_centered(d,(w/2,int(h*(.175 if portrait else .13))),eyebrow,head,(255,255,255,255))
    subtitle=str(q.get('note') or 'YOUR CHOICE. YOUR RULES.').upper()
    slines,sub=fit_lines(subtitle,int(w*.80),int(short*.029),min_size=max(12,int(short*.019)),max_lines=2)
    draw_centered(d,(w/2,int(h*(.249 if portrait else .235))),slines,sub,(215,223,245,220),spacing=int(short*.006))
    base.alpha_composite(typography)
    if portrait:
        gap=int(h*.026)
        card_w=int(w*.85);card_h=int(h*.236);card_x=(w-card_w)//2
        y1=int(h*.319);y2=y1+card_h+gap
        panel_a=opt_panel((card_w,card_h),p['left'],'A',q['a'],q.get('image_a'))
        panel_b=opt_panel((card_w,card_h),p['right'],'B',q['b'],q.get('image_b'))
        base.alpha_composite(panel_a,(card_x,y1));base.alpha_composite(panel_b,(card_x,y2))
        # VS medallion bridging two cards
        medallion=Image.new('RGBA',(w,h),(0,0,0,0)); d=ImageDraw.Draw(medallion,'RGBA')
        cx=w//2;cy=y2-gap//2;rad=int(w*.055)
        d.ellipse((cx-rad,cy-rad,cx+rad,cy+rad),fill=(22,18,48,255),outline=(255,255,255,160),width=3)
        vfont=font(int(w*.042),heavy=True)
        draw_centered(d,(cx,cy-vfont.size*.63),'VS',vfont,(255,255,255,255))
        footer_top=int(h*.863)
    else:
        gap=int(w*.026)
        card_w=int(w*.39);card_h=int(h*.43)
        x1=int(w*.09);x2=x1+card_w+gap
        y1=int(h*.32)
        base.alpha_composite(opt_panel((card_w,card_h),p['left'],'A',q['a'],q.get('image_a'),True),(x1,y1))
        base.alpha_composite(opt_panel((card_w,card_h),p['right'],'B',q['b'],q.get('image_b'),True),(x2,y1))
        medallion=Image.new('RGBA',(w,h),(0,0,0,0)); d=ImageDraw.Draw(medallion,'RGBA')
        cx=w//2;cy=y1+card_h//2;rad=int(h*.066)
        d.ellipse((cx-rad,cy-rad,cx+rad,cy+rad),fill=(22,18,48,255),outline=(255,255,255,160),width=3)
        vfont=font(int(h*.063),heavy=True)
        draw_centered(d,(cx,cy-vfont.size*.63),'VS',vfont,(255,255,255,255))
        footer_top=int(h*.84)
    base.alpha_composite(medallion)
    d=ImageDraw.Draw(base,'RGBA')
    footfont=font(int(short*.033),bold=True)
    draw_centered(d,(w/2,footer_top),'COMMENT A OR B  ↓',footfont,(235,241,251,248))
    d.text((margin,h-int(h*.054)), 'CHOICEFORGE  /  EDITION 01',font=font(int(short*.017),bold=True),fill=(200,214,240,150))
    return base.convert('RGB')


def splash_scene(cfg,w,h,ending=False):
    p=THEMES[cfg['theme']]; b=bg_image(w,h,cfg['theme'],cfg.get('background_image'),seed=43)
    d=ImageDraw.Draw(b,'RGBA')
    short=min(w,h)
    if ending:
        tag='YOU MADE IT.'
        headline='WHAT DID\nYOU PICK?'
        caption='DROP YOUR PICKS IN THE COMMENTS'
    else:
        tag='THE ULTIMATE CHOICE GAME'
        headline=cfg.get('title','WOULD YOU RATHER?').upper()
        caption=f'{len(cfg["questions"])} ROUNDS   ·   2 CHOICES   ·   NO WRONG ANSWERS'
    f=font(int(short*.04),bold=True)
    draw_centered(d,(w/2,int(h*.20)),tag,f,(*p['left'],255))
    title_size=int(short*(.115 if w<h else .085))
    lines=[]
    for part in headline.split('\n'):
        a,_=fit_lines(part,int(w*.86),title_size,min_size=int(short*.05),max_lines=5)
        lines.extend(a)
    hf=font(title_size,heavy=True)
    lines2,hf=fit_lines(' '.join(lines),int(w*.88),title_size,min_size=int(short*.045),max_lines=5)
    lh=hf.size*1.2
    yy=h*.39-len(lines2)*lh/2
    for i,line in enumerate(lines2):
        bb=d.textbbox((0,0),line,font=hf);tw=bb[2]-bb[0]
        d.text(((w-tw)/2,yy+i*lh),line,font=hf,fill=(255,255,255,255))
    sf=font(int(short*.032),bold=True)
    draw_centered(d,(w/2,int(h*.67)),caption,sf,(232,233,255,255))
    pill(d,(int(w*.2),int(h*.77),int(w*.8),int(h*.83)),'LET\'S GO' if not ending else 'PLAY AGAIN',font(int(short*.034),bold=True),(15,17,35),fill=(*p['left'],255),radius=round(short*.025))
    return b.convert('RGB')


def make_sfx(path:Path, durations:list[float], intro:float, outro:float, sample_rate=44100):
    total=int((intro+sum(durations)+outro+.1)*sample_rate)
    a=np.zeros(total,dtype=np.float32)
    def sine(t, duration, freq, amp):
        start=int(t*sample_rate); n=int(duration*sample_rate)
        if n<=0 or start+n>total:return
        x=np.arange(n,dtype=np.float32)/sample_rate
        env=np.minimum(1,x/.02)*np.maximum(0,1-x/duration)**2
        # A light generated digital bell, not copyrighted sample
        a[start:start+n] += (np.sin(2*np.pi*freq*x)+.30*np.sin(2*np.pi*freq*2*x))*amp*env
    sine(.15,.36,780,.10)
    t=intro
    for dur in durations:
        sine(t+.10,.12,930,.08)
        sine(t+dur-.49,.26,1175,.075)
        t+=dur
    sine(t+.17,.47,880,.10)
    a=np.clip(a,-.95,.95)
    with wave.open(str(path),'wb') as wf:
        wf.setnchannels(1);wf.setsampwidth(2);wf.setframerate(sample_rate)
        wf.writeframes((a*32767).astype('<i2').tobytes())


def choose_music(library:Path,mood:list[str]):
    """Match only user-provided tracks with known license metadata; never scrape YouTube."""
    manifest=library/'tracks.json'
    if not manifest.exists():return None,None
    try: items=json.loads(manifest.read_text(encoding='utf-8'))
    except (ValueError,OSError):return None,None
    scored=[]
    for item in items:
        file=library/Path(item.get('file','')).name
        if not file.is_file():continue
        if not item.get('license') or not item.get('source'):
            continue
        if 'attribution required' in str(item.get('license','')).lower() and not item.get('attribution'):
            continue
        tags={str(x).lower() for x in item.get('moods',[])}
        score=len(tags.intersection(mood))
        scored.append((score, str(file), item))
    if not scored:return None,None
    scored.sort(key=lambda x:(-x[0],x[1]))
    return scored[0][1],scored[0][2]


def write_audio_manifest(out:Path, music_meta:Optional[dict], music_path:Optional[str]):
    if music_path is None:
        txt='No background music was added. Generated synthetic sound effects are original.\n'
    elif music_meta:
        txt=f"MUSIC TRACK: {music_meta.get('title','Unknown')}\nSOURCE: {music_meta.get('source')}\nLICENSE: {music_meta.get('license')}\nATTRIBUTION: {music_meta.get('attribution','No attribution supplied — verify license')}\n\nKeep a local record of your permission/license.\n"
    else:
        txt=f"CUSTOM USER MUSIC: {Path(music_path).name}\nRights not verified by this tool. You must own or license the music for this video.\n"
    (out/'MUSIC_LICENSE.txt').write_text(txt,encoding='utf-8')


def _ffmpeg_available():
    if not shutil.which('ffmpeg'):raise RuntimeError('FFmpeg not found on PATH. Install FFmpeg first.')


def render(cfg:dict, output:str|Path, quality='Preview', music_mode='none', custom_music:Optional[str]=None, include_sfx=True, callback:Optional[Callable[[float,str],None]]=None, voice_synthesizer=None):
    """Export full project, no-music clean MP4, music-mixed MP4, thumbnail and timestamp sheet."""
    validate_config(cfg);_ffmpeg_available()
    if not shutil.which('ffprobe'):raise RuntimeError('FFprobe is required')
    output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=True)
    fps=int(cfg.get('fps',24)); fps=24 if fps not in (24,30) else fps
    w,h=canvas_dims(cfg['aspect'],quality)
    duration=float(cfg['seconds_per_question']); intro=float(cfg.get('intro_seconds',1.5));outro=float(cfg.get('outro_seconds',1.5))
    if not 0.5<=intro<=10 or not 0.5<=outro<=10:raise ValueError('Intro/outro must be 0.5–10 seconds')
    base_durations=[intro]+[duration]*len(cfg['questions'])+[outro]
    if callback:callback(0.01,'Preparing voiceover (each phrase is separately paced)')
    measured,narration_path,narration_cues=render_voice(cfg,output,fps,base_durations,synthesizer=voice_synthesizer)
    counts=[max(1,math.ceil(x*fps)) for x in measured]
    durations=[c/fps for c in counts];full_duration=sum(durations)
    scenes=[splash_scene(cfg,w,h)]
    scenes.extend(round_scene(cfg,w,h,i) for i in range(len(cfg['questions'])))
    scenes.append(splash_scene(cfg,w,h,ending=True))
    bases=[np.ascontiguousarray(np.asarray(img)[:,:,::-1]) for img in scenes]
    # Thumbnail from introductory scene, separate high-quality PNG.
    scenes[0].save(output/'thumbnail.png',optimize=True)
    (output/'project.json').write_text(json.dumps(cfg,indent=2,ensure_ascii=False),encoding='utf-8')
    timeline=[];start=0.
    for i,count in enumerate(counts):
        typ='INTRO' if i==0 else 'OUTRO' if i==len(counts)-1 else 'ROUND'
        desc=cfg['title'] if typ!='ROUND' else f"A: {cfg['questions'][i-1]['a']} | B: {cfg['questions'][i-1]['b']}"
        timeline.append((typ,f'{start:.2f}',f'{start+count/fps:.2f}',desc))
        start+=count/fps
    with (output/'timeline.csv').open('w',newline='',encoding='utf-8') as fp:
        writer=csv.writer(fp);writer.writerow(['type','start_seconds','end_seconds','text']);writer.writerows(timeline)
    (output/'description.txt').write_text(f"{cfg['title']}\n\n{len(cfg['questions'])} original choices. Pick A or B in each round and tell us your favorites in the comments!\n\n#WouldYouRather #ThisOrThat #InteractiveQuiz\n\nMusic credits (if required): add license attribution from MUSIC_LICENSE.txt\n",encoding='utf-8')
    silent=output/'video_clean.mp4'
    cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-vcodec','rawvideo','-s',f'{w}x{h}','-pix_fmt','bgr24','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','veryfast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(silent)]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    total_frames=sum(counts); done=0
    try:
        for si,nframes in enumerate(counts):
            src=bases[si];previous=bases[si-1] if si else None
            is_round=0<si<len(counts)-1
            for f in range(nframes):
                t=f/fps; dur=nframes/fps
                # Reuse the full-res scene; copy only on frames with animation.
                frame=src.copy()
                motion=cfg.get('motion','Gentle')
                if motion!='Off':
                    # Very subtle kinetic framing; animated even if timer/SFX are off.
                    strength=.012 if motion=='Gentle' else .030
                    phase=(t/max(dur,.01))*2*math.pi
                    zoom=1+strength*(.6+.4*math.sin(phase-.6))
                    dx=w*strength*.17*math.sin(phase*.73+si*.66)
                    dy=h*strength*.11*math.cos(phase*.83+si*.42)
                    affine=cv2.getRotationMatrix2D((w/2,h/2),0,zoom)
                    affine[0,2]+=dx;affine[1,2]+=dy
                    frame=cv2.warpAffine(frame,affine,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
                if is_round:
                    # Accent progress bar; color matches theme; interactive countdown.
                    accent=THEMES[cfg['theme']]['left']
                    left=int(w*.13);right=int(w*.87);yb=int(h*(.927 if w<h else .947))
                    bh=max(8,int(min(w,h)*.012)); ww=right-left
                    cv2.rectangle(frame,(left,yb),(right,yb+bh),(58,55,67),-1,cv2.LINE_AA)
                    progress=max(0,1-t/dur)
                    pw=max(0,int(ww*progress))
                    if pw:cv2.rectangle(frame,(left,yb),(left+pw,yb+bh),tuple(reversed(accent)),-1,cv2.LINE_AA)
                    # Timer badge. Displays 5..1, visible from early in the round.
                    n=max(1,math.ceil(dur-t))
                    boxsize=int(min(w,h)*.09)
                    cx=int(w*.84);cy=int(h*(.118 if w<h else .10))
                    overlay=frame.copy()
                    cv2.circle(overlay,(cx,cy),boxsize//2,(255,255,255),-1,cv2.LINE_AA)
                    cv2.addWeighted(overlay,.12,frame,.88,0,frame)
                    cv2.putText(frame,str(n),(cx-int(boxsize*.20),cy+int(boxsize*.17)),cv2.FONT_HERSHEY_DUPLEX,boxsize/77,(255,255,255),max(2,int(boxsize/22)),cv2.LINE_AA)
                    if t>dur-.55:
                        # Final visual accent to create a satisfying ending
                        height=int(min(w,h)*.017)
                        cv2.rectangle(frame,(int(w*.33),int(h*.80)),(int(w*.67),int(h*.80)+height),tuple(reversed(accent)),-1,cv2.LINE_AA)
                # Rapid crossfade on every scene change.
                faden=min(8,nframes//3)
                if previous is not None and f<faden and faden>1:
                    cv2.addWeighted(frame,(f+1)/faden,previous,1-(f+1)/faden,0,frame)
                # Slight overall fade-in / fade-out on the full video.
                if si==0 and f<7:frame=cv2.convertScaleAbs(frame,alpha=(f+1)/7,beta=0)
                if si==len(counts)-1 and f>=nframes-8:frame=cv2.convertScaleAbs(frame,alpha=max(.01,(nframes-f)/8),beta=0)
                proc.stdin.write(frame.tobytes())
                done+=1
                if callback and done%max(1,fps*2)==0:callback(.04+.91*done/total_frames, f'Rendering {done}/{total_frames} frames')
        proc.stdin.close()
        rc=proc.wait()
        if rc:raise RuntimeError('FFmpeg video encoding failed: '+proc.stderr.read().decode('utf-8','replace')[-1300:])
    except (Exception,KeyboardInterrupt):
        try:proc.kill()
        except Exception:pass
        try:proc.wait(timeout=4)
        except Exception:pass
        raise
    finally:
        if proc.stderr:proc.stderr.close()
    music_path=None;metadata=None
    if music_mode == 'custom' and custom_music:
        if not Path(custom_music).is_file():raise ValueError('Selected custom music file not found')
        music_path=str(Path(custom_music).resolve())
    elif music_mode == 'auto':
        music_path,metadata=choose_music(ROOT/'assets'/'music',THEMES[cfg['theme']]['mood'])
    if metadata and metadata.get('attribution'):
        with (output/'description.txt').open('a',encoding='utf-8') as description:
            description.write('\nMusic attribution: '+metadata['attribution']+'\n')
    sfx_path=output/'original_sfx.wav'
    if include_sfx:make_sfx(sfx_path,durations[1:-1],durations[0],durations[-1])
    write_audio_manifest(output,metadata,music_path)
    final=output/'video_final.mp4'
    if music_path or include_sfx or narration_path:
        acmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(silent)]
        tracks=[];sources=[]
        input_idx=1
        if music_path:
            acmd += ['-stream_loop','-1','-i',music_path]
            fadeout=max(0,full_duration-.85)
            vol='0.11' if narration_path else '0.23'
            tracks.append(f'[{input_idx}:a]volume={vol},atrim=0:{full_duration:.4f},asetpts=PTS-STARTPTS,afade=t=in:st=0:d=0.50,afade=t=out:st={fadeout:.3f}:d=0.80[bg]')
            sources.append('[bg]');input_idx+=1
        if include_sfx:
            acmd+=['-i',str(sfx_path)]
            tracks.append(f'[{input_idx}:a]volume={"0.36" if narration_path else "0.75"},aresample=44100[sfx]')
            sources.append('[sfx]');input_idx+=1
        if narration_path:
            acmd+=['-i',str(narration_path)]
            # Gentle compression makes the voice easier to hear at mobile volume.
            tracks.append(f'[{input_idx}:a]aresample=44100,highpass=f=95,acompressor=threshold=0.13:ratio=2.5:attack=5:release=140,volume=1.25[vox]')
            sources.append('[vox]')
        if len(sources)>1:
            tracks.append(''.join(sources)+f'amix=inputs={len(sources)}:duration=longest:normalize=0,alimiter=limit=0.94[a]')
            aud='[a]'
        else:
            tracks.append(sources[0]+'anull[a]');aud='[a]'
        acmd += ['-filter_complex',';'.join(tracks),'-map','0:v:0','-map',aud,'-c:v','copy','-c:a','aac','-b:a','192k','-t',f'{full_duration:.4f}','-movflags','+faststart',str(final)]
        res=subprocess.run(acmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        if res.returncode:raise RuntimeError('Audio mixing failed: '+res.stderr[-1800:])
    else:
        shutil.copyfile(silent,final)
    if callback: callback(1.0,'Done')
    return {'video':str(final),'clean':str(silent),'thumbnail':str(output/'thumbnail.png'),'project':str(output/'project.json'),'timeline':str(output/'timeline.csv'),'music_license':str(output/'MUSIC_LICENSE.txt'),'duration':round(full_duration,2),'resolution':f'{w}x{h}','audio_selected':music_path,'narration':str(narration_path) if narration_path else None,'voice_segments':len(narration_cues)}


def read_project(path:str|Path):
    path=Path(path).resolve()
    cfg=json.loads(path.read_text(encoding='utf-8'))
    for key in ('background_image',):
        if cfg.get(key) and not Path(cfg[key]).is_absolute(): cfg[key]=str((path.parent/cfg[key]).resolve())
    for q in cfg.get('questions',[]):
        for key in ('image_a','image_b'):
            if q.get(key) and not Path(q[key]).is_absolute(): q[key]=str((path.parent/q[key]).resolve())
    validate_config(cfg)
    return cfg


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Render ChoiceForge would-you-rather videos')
    parser.add_argument('--project',default='examples/default_project.json')
    parser.add_argument('--output',default='outputs/my_video')
    parser.add_argument('--quality',choices=['Preview','Full HD'],default='Preview')
    parser.add_argument('--music-mode',choices=['none','auto','custom'],default='none')
    parser.add_argument('--music-file')
    parser.add_argument('--no-sfx',action='store_true')
    args=parser.parse_args()
    print(json.dumps(render(read_project(args.project),args.output,quality=args.quality,music_mode=args.music_mode,custom_music=args.music_file,include_sfx=not args.no_sfx,callback=lambda x,msg:print(f'\r{x:.0%} {msg}',end='',flush=True)),indent=2))
