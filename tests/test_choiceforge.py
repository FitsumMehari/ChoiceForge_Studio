import json
import shutil
import subprocess
from pathlib import Path

import pytest

from renderer import (
    THEMES, choose_music, canvas_dims, read_project, render,
    round_scene, validate_config,
)


def project():
    return {
        'title':'Two impossible choices', 'format':'Would You Rather',
        'theme':'Electric Midnight','aspect':'9:16','fps':24,
        'intro_seconds':.5,'outro_seconds':.5,'seconds_per_question':2,
        'questions':[{'a':'Learn to fly','b':'Travel through time'}],
    }


def test_validate_rejects_empty_option():
    p=project();p['questions'][0]['b']=''
    with pytest.raises(ValueError):validate_config(p)


def test_invalid_dimensions():
    p=project();p['aspect']='1:1'
    with pytest.raises(ValueError):validate_config(p)


def test_portrait_and_landscape():
    p=project()
    a=round_scene(p,540,960,0)
    assert a.size==(540,960)
    p['aspect']='16:9'
    b=round_scene(p,960,540,0)
    assert b.size==(960,540)


def test_auto_music_chooses_only_documented(tmp_path):
    (tmp_path/'a.mp3').write_bytes(b'dummy')
    (tmp_path/'b.mp3').write_bytes(b'dummy')
    songs=[{'file':'a.mp3','title':'A','source':'YouTube Audio Library','license':'Audio Library','moods':['energetic']},
           {'file':'b.mp3','title':'B','moods':['energetic']}]
    (tmp_path/'tracks.json').write_text(json.dumps(songs))
    path,meta=choose_music(tmp_path,['energetic'])
    assert Path(path).name=='a.mp3'
    assert meta['title']=='A'

@pytest.mark.skipif(not shutil.which('ffmpeg'),reason='Requires FFmpeg')
def test_integration_render(tmp_path):
    result=render(project(),tmp_path/'output',quality='Preview',include_sfx=True)
    assert Path(result['video']).exists()
    assert Path(result['clean']).exists()
    assert Path(result['thumbnail']).exists()
    probe=subprocess.run(['ffprobe','-v','error','-show_entries','stream=width,height,codec_name','-of','json',result['video']],capture_output=True,text=True,check=True)
    streams=json.loads(probe.stdout)['streams']
    assert any(s.get('width')==540 and s.get('height')==960 and s.get('codec_name')=='h264' for s in streams)
    assert any(s.get('codec_name')=='aac' for s in streams)
