"""Cross-platform unit/integration tests; never depend on internet or paid APIs."""
import json
import shutil
import subprocess
import sys
import types
import wave
from pathlib import Path

import cv2
import numpy as np
import pytest

from app import build_cfg, package_output, preview_voice
from narration import (
    RATE, VOICE_PRESETS, decode_audio, default_narration, render_voice,
    script_for_scenes, voice_clip, voice_settings, write_wav,
)
from renderer import render, validate_config


def example(aspect='9:16', motion='Gentle', voice=True):
    return {
        'title':'Real voice timing test', 'format':'Would You Rather',
        'aspect':aspect,'theme':'Candy Pop','fps':24,
        'seconds_per_question':2.,'intro_seconds':.5,'outro_seconds':.5,
        'motion':motion,'questions':[{'a':'Fly over the city','b':'Swim with dolphins'}],
        'narration':{'enabled':voice, 'provider':'kokoro','voice':'af_heart',
                     'rate':8,'pitch':0,'pause':.35,'thinking_seconds':1.2, 'cache':False}
    }


def fake_tts(text,path,settings,key):
    # Grounded/synthetic speech stand-in. Writes a 0.25s audio clip in either WAV or MP3 format.
    samples=np.sin(np.arange(11025)*2*np.pi*390/RATE).astype(np.float32)*.22
    if Path(path).suffix=='.wav':
        write_wav(Path(path),samples)
    else:
        proc=subprocess.run(['ffmpeg','-v','error','-y','-f','f32le','-ar',str(RATE),
                             '-ac','1','-i','pipe:0','-f','mp3',str(path)],
                             input=samples.tobytes(),capture_output=True)
        assert proc.returncode==0,proc.stderr.decode()


def ffprobe(path):
    proc=subprocess.run(['ffprobe','-v','error','-show_entries','format=duration:stream=codec_type,codec_name,width,height',
                         '-of','json',str(path)],capture_output=True,text=True,check=True)
    return json.loads(proc.stdout)


def test_voice_presets_valid():
    assert len(VOICE_PRESETS)>=9
    assert VOICE_PRESETS['US · Heart (woman)']=='af_heart'
    assert VOICE_PRESETS['UK · George (man)']=='bm_george'
    assert default_narration()['provider']=='kokoro'


@pytest.mark.parametrize('field,value', [('provider','unknown'),('rate',100),('rate',-99),('pause',-2),('pause',2),('thinking_seconds',10),('pitch',45)])
def test_invalid_narration_settings(field,value):
    p=example();p['narration'][field]=value
    with pytest.raises(ValueError):validate_config(p)


def test_bad_motion():
    p=example();p['motion']='Spin wildly'
    with pytest.raises(ValueError,match='Motion'):validate_config(p)


@pytest.mark.parametrize('format_,prefix', [('Would You Rather','Would you rather'),('Pick One','Pick just one:'),('This or That','This or that:')])
def test_script_modes(format_,prefix):
    p=example();p['format']=format_
    scenes=script_for_scenes(p,voice_settings(p))
    assert len(scenes)==3
    assert scenes[1][0][1].startswith(prefix)
    assert len(scenes[1])==1
    assert scenes[1][0][0]=='question'
    assert 'or' in scenes[1][0][1].lower()


@pytest.mark.skipif(not shutil.which('ffmpeg'),reason='FFmpeg required')
def test_auto_timing_and_caption_cues(tmp_path):
    p=example()
    p['narration']['thinking_seconds']=2.2
    minimum=[.5,2.,.5]
    durations,path,reports=render_voice(p,tmp_path/'video',24,minimum,synthesizer=fake_tts)
    assert path.exists() and len(reports)==3
    assert durations[1]>minimum[1] # words, pauses, thinking period all respected
    assert durations[1]*24==round(durations[1]*24)
    for row in reports:
        assert row['start_seconds']>=0
        assert row['end_seconds']<=sum(durations)+.001
    assert reports[1]['start_seconds']>=durations[0]
    assert reports[-1]['start_seconds']>durations[0]+durations[1]
    content=(tmp_path/'video'/'narration.srt').read_text()
    assert ' --> ' in content and 'Fly over the city' in content
    samples=decode_audio(path)
    assert np.max(np.abs(samples))>.01
    assert len(samples)/RATE>=sum(durations)
    assert (tmp_path/'video'/'narration_script.txt').is_file()


@pytest.mark.skipif(not shutil.which('ffmpeg'),reason='FFmpeg required')
def test_no_truncation_when_long_phrase(tmp_path):
    p=example();p['questions'][0]['a']='An extraordinary once in a lifetime adventure far beyond the universe';p['narration']['thinking_seconds']=3
    durations,_,reports=render_voice(p,tmp_path/'long',24,[.5,2.,.5],synthesizer=fake_tts)
    assert durations[1]>2
    p['narration']['auto_extend']=False
    with pytest.raises(ValueError,match='needs'):
        render_voice(p,tmp_path/'blocked',24,[.5,2.,.5],synthesizer=fake_tts)


@pytest.mark.skipif(not shutil.which('ffmpeg'),reason='FFmpeg required')
def test_disabled_narration_is_offline(tmp_path):
    p=example(voice=False)
    durations,path,reports=render_voice(p,tmp_path,24,[1,2,1])
    assert durations==[1,2,1] and path is None and reports==[]


@pytest.mark.skipif(not shutil.which('ffmpeg'),reason='FFmpeg required')
@pytest.mark.parametrize('aspect,dimensions', [('9:16',(540,960)),('16:9',(960,540))])
def test_video_encode_with_voice_and_motion(tmp_path,aspect,dimensions):
    p=example(aspect=aspect)
    result=render(p,tmp_path/aspect.replace(':','_'),quality='Preview',include_sfx=True,voice_synthesizer=fake_tts)
    meta=ffprobe(result['video'])
    video=next(s for s in meta['streams'] if s['codec_type']=='video')
    audio=next(s for s in meta['streams'] if s['codec_type']=='audio')
    assert (video['width'],video['height'])==dimensions
    assert video['codec_name']=='h264' and audio['codec_name']=='aac'
    assert float(meta['format']['duration'])>=result['duration']-.1
    assert Path(result['narration']).exists()
    assert result['voice_segments']==3
    clean=ffprobe(result['clean'])
    assert len([s for s in clean['streams'] if s['codec_type']=='audio'])==0
    assert len([s for s in meta['streams'] if s['codec_type']=='audio'])==1
    assert (Path(result['video']).parent/'narration.srt').exists()
    package=Path(package_output(Path(result['video']).parent,p,None))
    import zipfile
    with zipfile.ZipFile(package) as zipf:
        names=zipf.namelist()
        assert 'narration.wav' in names and 'narration.srt' in names
        assert 'video_final.mp4' in names and 'voice_clips/scene_01_line_00.wav' in names
        assert 'project_portable.json' in names
        assert not any('secret' in n for n in names)


@pytest.mark.skipif(not shutil.which('ffmpeg'),reason='FFmpeg required')
def test_loudness_and_music_with_voice(tmp_path):
    song=tmp_path/'licensed.wav'
    write_wav(song,(np.sin(np.arange(RATE)*2*np.pi*90/RATE)*.22).astype(np.float32))
    p=example()
    r=render(p,tmp_path/'with_music',quality='Preview',music_mode='custom',custom_music=str(song),include_sfx=True,voice_synthesizer=fake_tts)
    audio=decode_audio(Path(r['video']))
    assert np.max(abs(audio))>.02
    assert np.max(abs(audio))<=1
    assert 'CUSTOM USER MUSIC' in Path(r['music_license']).read_text()


@pytest.mark.skipif(not shutil.which('ffmpeg'),reason='FFmpeg required')
def test_motion_changes_visuals(tmp_path):
    p=example(motion='Gentle',voice=False)
    p['seconds_per_question']=3
    r=render(p,tmp_path/'moving',quality='Preview',include_sfx=False)
    cap=cv2.VideoCapture(r['video'])
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC,2800); ok1,frame1=cap.read()
        cap.set(cv2.CAP_PROP_POS_MSEC,3500); ok2,frame2=cap.read()
        assert ok1 and ok2
        # Compare upper/lower background, not just timer badge/countdown.
        assert np.mean(abs(frame1[:80].astype(float)-frame2[:80].astype(float)))>.07
    finally:cap.release()


def test_secrets_not_added_to_config():
    p=build_cfg('Video','Would You Rather','Candy Pop','9:16',5,1,1,[['Choose time','Choose space','', '', '']],None)
    assert not any('api_key' in x for x in p)


def test_no_paid_or_robotic_providers(tmp_path):
    opts={**default_narration(),'provider':'elevenlabs','enabled':True}
    with pytest.raises(ValueError,match='Unknown narration provider'):
        voice_settings({'narration':opts})
    opts['provider']='offline'
    with pytest.raises(ValueError,match='Only free Kokoro'):
        voice_clip('Hello',tmp_path/'robot.wav',opts)


def test_interface_loads():
    import app
    assert app.demo is not None and len(app.demo.blocks)>30


def test_refuses_robotic_voice_for_production():
    opts={**default_narration(),'enabled':True,'provider':'offline'}
    with pytest.raises(ValueError,match='Unknown narration provider'):
        voice_settings({'narration':opts})


def test_legacy_segmented_mode_still_available():
    project=example()
    project['narration']['delivery_mode']='segmented'
    lines=script_for_scenes(project,voice_settings(project))
    assert len(lines[1])==4
    assert lines[1][0]==('setup','Would you rather')


def test_local_voice_requires_verified_model(monkeypatch,tmp_path):
    import free_voice
    monkeypatch.setattr(free_voice,'model_status',lambda: {'model':False,'voices':False})
    with pytest.raises(RuntimeError,match='not installed'):
        free_voice.synthesize('Hello',tmp_path/'test.wav')


def test_local_neural_voice_fake_model(monkeypatch,tmp_path):
    import free_voice
    import sys, types
    calls=[]
    class MockVoices:
        def close(self): calls.append('close')
    class FakeKokoro:
        def __init__(self,model,voices):
            calls.append((model,voices))
            self.voices=MockVoices()
        def create(self,text,voice,speed,lang):
            calls.append((text,voice,speed,lang))
            samples=np.sin(np.linspace(0,60,2400)).astype(np.float32)*.1
            return samples,24000
    monkeypatch.setattr(free_voice,'model_status',lambda: {'model':True,'voices':True})
    monkeypatch.setitem(sys.modules,'kokoro_onnx',types.SimpleNamespace(Kokoro=FakeKokoro))
    for voice,accent in [('af_heart','en-us'),('bm_george','en-gb')]:
        path=tmp_path/f'{voice}.wav'
        free_voice.synthesize('Test voice',path,voice=voice,speed=.96)
        with wave.open(str(path)) as w:
            assert w.getframerate()==24000 and w.getnframes()==2400
        assert calls[-1]==('Test voice',voice,.96,accent)


def test_model_installer_verifies_and_is_idempotent(monkeypatch,tmp_path):
    import free_voice
    import io,hashlib
    monkeypatch.setattr(free_voice,'MODEL_DIR',tmp_path)
    files={'one.onnx':b'sample-ONNX','voices.bin':b'file-of-voices'}
    monkeypatch.setattr(free_voice,'FILES',{k:hashlib.sha256(v).hexdigest() for k,v in files.items()})
    requests=[]
    class Reply(io.BytesIO):
        def __enter__(self):return self
        def __exit__(self,*args):pass
    def fakeopen(req, timeout=40):
        name=req.full_url.rsplit('/',1)[-1];requests.append(name)
        return Reply(files[name])
    monkeypatch.setattr(free_voice.urllib.request,'urlopen',fakeopen)
    assert sorted(free_voice.install_models())==sorted(files)
    assert all(free_voice.model_status().values())
    assert free_voice.install_models()==[]
    assert sorted(requests)==sorted(files)


def test_model_download_rejects_corrupted_file(monkeypatch,tmp_path):
    import free_voice,io
    monkeypatch.setattr(free_voice,'MODEL_DIR',tmp_path)
    monkeypatch.setattr(free_voice,'FILES',{'bad.onnx':'0'*64})
    class Reply(io.BytesIO):
        def __enter__(self):return self
        def __exit__(self,*args):pass
    monkeypatch.setattr(free_voice.urllib.request,'urlopen',lambda req,timeout=40:Reply(b'corrupted'))
    with pytest.raises(RuntimeError,match='Checksum mismatch'):
        free_voice.install_models()
    assert not (tmp_path/'bad.onnx').exists()
    assert not list(tmp_path.glob('*.part'))


def test_energetic_pacing_adjusts_real_tts_rate(tmp_path):
    project=example();project['narration']['energy']='Energetic game-show host'
    captured=[]
    def capture(text,path,opts,key):
        captured.append(opts['rate'])
        fake_tts(text,path,opts,key)
    render_voice(project,tmp_path/'energy',24,[1,2,1],synthesizer=capture)
    assert len(captured)==3
    assert captured[1]==13  # project baseline +8, game-show +6, question -1

def test_zero_cost_ui_exposes_no_paid_key_fields():
    from app import preview_voice, run_render
    assert 'eleven' not in preview_voice.__code__.co_varnames
    assert 'api_key' not in run_render.__code__.co_varnames
