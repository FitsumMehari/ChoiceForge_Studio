"""ChoiceForge free natural-voice pipeline powered by local Kokoro ONNX.

Every production voice is neural. No paid service, account, or robotic fallback.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RATE = 44100
from free_voice import VOICES_FREE

DEFAULT_VOICE = 'af_heart'
VOICE_PRESETS = VOICES_FREE


@dataclass
class SpeechClip:
    scene: int
    text: str
    wav: np.ndarray
    lead: float
    tail: float
    role: str


def default_narration():
    return {'enabled': False, 'provider': 'kokoro', 'voice': DEFAULT_VOICE,
            'rate': 0, 'pitch': 0, 'energy': 'Warm conversational',
            'pause': 0.45, 'thinking_seconds': 2.0, 'auto_extend': True,
            'quality_mode': 'natural_only', 'delivery_mode': 'conversational',
            'intro_text': 'Ready for an impossible choice? Let’s play!',
            'outro_text': 'Which choices were the hardest? Tell me below!'}


def voice_settings(config: dict) -> dict:
    opts={**default_narration(), **(config.get('narration') or {})}
    if opts['provider'] != 'kokoro':
        raise ValueError('Unknown narration provider')
    if opts.get('delivery_mode') not in ('conversational', 'segmented'):
        raise ValueError('Unknown narration delivery mode')
    if opts.get('quality_mode') not in ('natural_only', 'diagnostic'):
        raise ValueError('Unknown narration quality mode')
    if not -25 <= int(opts['rate']) <= 25:
        raise ValueError('Voice speed adjustment must be between -25 and 25 percent')
    if not -30 <= int(opts['pitch']) <= 30:
        raise ValueError('Pitch adjustment must be between -30 and 30 Hz')
    if not 0 <= float(opts['pause']) <= 1.5:
        raise ValueError('Pause must be between 0 and 1.5 seconds')
    if not .5 <= float(opts['thinking_seconds']) <= 8:
        raise ValueError('Thinking period must be between .5 and 8 seconds')
    if len(str(opts['voice']))>120:raise ValueError('Voice name too long')
    for k in ('intro_text','outro_text'):
        if len(str(opts.get(k,'')))>300: raise ValueError(f'{k} must be shorter than 300 characters')
    return opts


def script_for_scenes(cfg: dict, v: dict) -> list[list[tuple[str,str]]]:
    """Connected sentences avoid the disjoint tone of separate A/B TTS calls.

    For old projects, segmented mode preserves the previous line-by-line timing.
    """
    intro=str(v.get('intro_text','')).strip()
    outro=str(v.get('outro_text','')).strip()
    scenes=[[('intro', intro)] if intro else []]
    for index, q in enumerate(cfg['questions']):
        a=str(q['a']).strip().rstrip('.,?!')
        b=str(q['b']).strip().rstrip('.,?!')
        if v.get('delivery_mode') == 'conversational':
            fmt=cfg.get('format')
            if fmt == 'Would You Rather':
                question=f'Would you rather {a}... or {b}?'
            elif fmt == 'Pick One':
                question=f'Pick just one: {a}... or {b}?'
            else:
                question=f'This or that: {a}... or {b}?'
            # A single synthesis call keeps both options in one natural prosodic arc.
            scenes.append([('question', question)])
        elif cfg.get('format') == 'Would You Rather':
            scenes.append([('setup','Would you rather'),('a',a+'?'),('b','Or '+b+'?'),('prompt','Choose A, or B!')])
        elif cfg.get('format') == 'Pick One':
            scenes.append([('setup','Pick one.'),('a',a+'.'),('b','Or '+b+'.'),('prompt','Make your choice!')])
        else:
            scenes.append([('setup','This or that?'),('a',a+'.'),('b','Or '+b+'.'),('prompt','Which one wins?')])
    scenes.append([('outro',outro)] if outro else [])
    return scenes


def _run(command:list[str], *, timeout=90, input_bytes=None) -> bytes:
    try:
        res=subprocess.run(command, input=input_bytes,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout)
    except (OSError,subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f'Audio command failed or timed out: {command[0]}') from exc
    if res.returncode:
        raise RuntimeError('Audio processing failed: '+res.stderr.decode(errors='replace')[-700:])
    return res.stdout


def decode_audio(path:Path) -> np.ndarray:
    pcm=_run(['ffmpeg','-v','error','-i',str(path),'-vn','-ac','1','-ar',str(RATE),'-f','s16le','pipe:1'])
    if not pcm:raise RuntimeError('TTS provider returned empty audio')
    return np.frombuffer(pcm,dtype='<i2').astype(np.float32)/32768


def write_wav(path:Path, samples:np.ndarray):
    samples=np.clip(samples,-.96,.96)
    with wave.open(str(path),'wb') as wav:
        wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(RATE)
        wav.writeframes((samples*32767).astype('<i2').tobytes())


def voice_clip(text: str, output: Path, v: dict, key: str | None = None):
    """Always synthesize free neural speech locally, never use a robot voice."""
    if v['provider'] != 'kokoro':
        raise ValueError('Only free Kokoro local narration is supported.')
    from free_voice import synthesize
    speed = 1 + int(v['rate']) / 100
    synthesize(text, output, voice=v['voice'], speed=max(.70, min(1.30, speed)))
    if not output.exists() or output.stat().st_size < 100:
        raise RuntimeError('The free local narrator did not produce audio.')


def render_voice(cfg:dict, out:Path, fps:int, minimum_scene_durations:list[float], key:str|None=None, synthesizer=None):
    """Returns scene durations, narration WAV path, metadata. Uses actual audio lengths.

    Never speed/truncate audio to force it into the video; scene durations auto-grow.
    """
    v=voice_settings(cfg)
    if not v['enabled']:
        return list(minimum_scene_durations),None,[]
    scenes=script_for_scenes(cfg,v)
    if len(scenes)!=len(minimum_scene_durations):raise ValueError('Narration scene mismatch')
    audio_dir=out/'voice_clips';audio_dir.mkdir(exist_ok=True,parents=True)
    all_clips=[];durations=[];reports=[]
    if synthesizer is None:synthesizer=voice_clip
    gap=float(v['pause']);think=float(v['thinking_seconds'])
    for i,phrases in enumerate(scenes):
        generated=[]
        for j,(role,text) in enumerate(phrases):
            if not text.strip():continue
            extension='.wav'
            filename=audio_dir/f'scene_{i:02d}_line_{j:02d}{extension}'
            # Subtle phrase-by-phrase delivery changes: the hook is measured,
            # options are slightly more deliberate, CTA has extra energy.
            delivery=dict(v)
            role_rates={'intro':0,'outro':-2,'question':-1,'setup':-2,'a':-3,'b':-3,'prompt':6}
            styles={'Warm conversational':0, 'Energetic game-show host':6,
                    'Dramatic suspense':-5, 'Calm storytelling':-7}
            delivery['rate']=min(25,max(-25,int(v['rate']) + role_rates.get(role,0)
                                                       + styles.get(v.get('energy'),0)))
            delivery['pitch']=min(30,max(-30,int(v['pitch'])+(4 if role=='prompt' else 0)))
            # Voice clips are cached locally; no external services or secrets.
            cache_info=json.dumps({'text':text,'provider':delivery['provider'],
                'quality_mode':delivery.get('quality_mode'),'delivery_mode':delivery.get('delivery_mode'),
                'voice':delivery['voice'],'model':'kokoro-v1.0.int8', 'rate':delivery['rate'],
                'pitch':delivery['pitch'],'energy':delivery.get('energy')},sort_keys=True,ensure_ascii=False)
            cache_key=hashlib.sha256(cache_info.encode()).hexdigest()[:24]
            cache_dir=Path(__file__).resolve().parent/'cache'/'voice'
            # Caching can be disabled to avoid storing narration in shared environments.
            use_cache=bool(v.get('cache',True))
            cached=cache_dir/(cache_key+extension)
            if use_cache and cached.is_file() and cached.stat().st_size>200:
                shutil.copyfile(cached,filename)
            else:
                synthesizer(text,filename,delivery,key)
                if use_cache:
                    cache_dir.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(filename,cached)
            samples=decode_audio(filename)
            # Prevent unexpected absurd output duration from destroying short form.
            if not 0.05 < len(samples)/RATE < 60:
                raise RuntimeError(f'Narration clip {i}/{j} has unexpected duration')
            generated.append((role,text,samples))
        cursor=.12 if i else .20
        entries=[]
        for j,(role,text,samples) in enumerate(generated):
            # Segmented legacy mode has deliberately longer pauses; conversational
            # mode is a single natural voice clip per question.
            if j and i not in (0,len(scenes)-1):
                cursor+=gap*(1.3 if role=='b' else 1.0)
            lead=cursor
            cursor+=len(samples)/RATE
            entries.append(SpeechClip(i,text,samples,lead,0,role))
        extra=think if 0<i<len(scenes)-1 else .25
        required=cursor+extra
        lower=minimum_scene_durations[i]
        if required>lower and not v.get('auto_extend',True):
            raise ValueError(f'Round {i} needs {required:.1f}s to speak naturally; enable auto-extend or increase seconds per question')
        seconds=max(lower,required)
        seconds=math.ceil(seconds*fps)/fps
        durations.append(seconds);all_clips.append(entries)
        for e in entries:
            reports.append({'scene':i,'role':e.role,'text':e.text,'start_in_scene':round(e.lead,3),'duration':round(len(e.wav)/RATE,3)})
    elapsed=0.
    for scene_id,duration in enumerate(durations):
        for entry in reports:
            if entry['scene']==scene_id:
                entry['start_seconds']=round(elapsed+entry['start_in_scene'],3)
                entry['end_seconds']=round(elapsed+entry['start_in_scene']+entry['duration'],3)
        elapsed+=duration
    def srt_timestamp(time):
        milliseconds=max(0,round(time*1000))
        h,rem=divmod(milliseconds,3600000)
        m,rem=divmod(rem,60000)
        sec,ms=divmod(rem,1000)
        return f'{h:02d}:{m:02d}:{sec:02d},{ms:03d}'
    captions=''.join(f'{j}\n{srt_timestamp(e["start_seconds"])} --> {srt_timestamp(e["end_seconds"])}\n{e["text"]}\n\n' for j,e in enumerate(reports,1))
    (out/'narration.srt').write_text(captions,encoding='utf-8')
    total=math.ceil(sum(durations)*RATE)+int(RATE*.05)
    audio=np.zeros(total,dtype=np.float32)
    shift=0.
    for dur,entries in zip(durations,all_clips):
        for e in entries:
            idx=round((shift+e.lead)*RATE); end=min(idx+len(e.wav),len(audio))
            audio[idx:end]+=e.wav[:end-idx]*.89
        shift+=dur
    wav=out/'narration.wav';write_wav(wav,audio)
    (out/'narration_cues.json').write_text(json.dumps(reports,indent=2,ensure_ascii=False),encoding='utf-8')
    (out/'narration_script.txt').write_text('\n'.join(f"Scene {r['scene']} [{r['role']}]: {r['text']}" for r in reports)+'\n',encoding='utf-8')
    return durations,wav,reports
