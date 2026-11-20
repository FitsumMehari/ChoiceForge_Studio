"""ChoiceForge Studio local browser editor — run `python app.py`."""
from __future__ import annotations

import json
import os
import random
import re
import shutil
import time
import zipfile
from pathlib import Path

import gradio as gr

from renderer import DEFAULT_QUESTIONS, ROOT, THEMES, bg_image, canvas_dims, read_project, render, round_scene, splash_scene
from narration import VOICE_PRESETS, default_narration, voice_clip
from free_voice import model_status, install_models
from trivia_video import build_episodes, load_bank, render_series, resolve_zips, summary

OUTPUTS = ROOT / 'outputs'
OUTPUTS.mkdir(exist_ok=True)
BANK = {
    'Fantasy': [
        ('Read minds for five minutes each day','See 30 seconds into the future'),
        ('Own a flying house','Own a submarine you could live in'),
        ('Speak with animals','Talk to any historical figure for an hour'),
        ('Control the weather in your city','Never need to sleep again'),
        ('Travel between two planets instantly','Explore a new universe for one day'),
        ('Change your appearance whenever you want','Change your voice to sound like anyone'),
        ('Have a pet dragon','Have a friendly personal robot'),
        ('Live inside your favorite video game','Visit any fictional world for a weekend'),
    ],
    'Lifestyle': [
        ('Never wait in a queue again','Never sit in traffic again'),
        ('Always get the best seat at events','Always get upgraded when traveling'),
        ('Have a private chef for life','Have a personal travel planner for life'),
        ('Live in a cabin surrounded by forests','Live in a modern apartment above the clouds'),
        ('Be able to instantly master a new skill','Remember every book you read'),
        ('Get one free flight every month','Get one free hotel stay every month'),
        ('Have a 4-day workweek forever','Have 8 weeks of vacation every year'),
        ('Spend a year exploring the ocean','Spend a year traveling across continents'),
    ],
    'Technology': [
        ('Have an AI assistant that manages your chores','Have a robot that cooks every meal'),
        ('Own a computer that never slows down','Own a phone that never needs charging'),
        ('Be an expert in every programming language','Be an expert in every spoken language'),
        ('Have perfect internet everywhere','Have unlimited cloud storage forever'),
        ('Live in a smart house on a mountain','Live on a futuristic floating city'),
        ('See the future of technology in 2100','Travel back to experience the internet in 1995'),
        ('Invent a technology everyone uses','Build a game everyone loves'),
        ('Never lose a file again','Never forget a password again'),
    ],
    'Food & Travel': [
        ('Eat amazing sushi in Tokyo tonight','Eat fresh pasta in Rome tonight'),
        ('Explore the pyramids in Egypt','See the northern lights in Iceland'),
        ('Take a train trip through Europe','Drive the entire Pacific Coast Highway'),
        ('Have fresh fruit delivered daily','Have your favorite dessert delivered daily'),
        ('Live near a tropical beach','Live near snowy mountain peaks'),
        ('Go camping under the stars','Stay in a luxurious treehouse'),
        ('Have free coffee for life','Have free smoothies for life'),
        ('Visit every world capital','Visit every UNESCO natural wonder'),
    ],
}

def default_rows():
    return [[q['a'],q['b'],q.get('note',''),'', ''] for q in DEFAULT_QUESTIONS]


def rows_for(category, how_many, seed):
    rng=random.Random(int(seed))
    pool=BANK[category].copy();rng.shuffle(pool)
    n=max(1,min(20,int(how_many)))
    selected=[]
    while len(selected)<n:
        rng.shuffle(pool)
        for a,b in pool:
            if (a,b) not in selected:
                selected.append((a,b))
            if len(selected)>=n:break
        if len(selected)>=len(BANK[category]):break
    return [[a,b,'Which would you pick?','',''] for a,b in selected]




def convert_rows(data):
    if hasattr(data,'values'):data=data.values.tolist()
    if not isinstance(data,(list,tuple)): raise ValueError('Question table must contain rows')
    out=[]
    for row in data:
        if not row or len(row)<2:continue
        a,b=[str(v or '').strip() for v in row[:2]]
        if not a and not b:continue
        if not a or not b:raise ValueError('Each populated row needs both option A and option B')
        q={'a':a,'b':b}
        if len(row)>2 and str(row[2] or '').strip():q['note']=str(row[2]).strip()
        if len(row)>3 and str(row[3] or '').strip():q['image_a']=str(row[3]).strip()
        if len(row)>4 and str(row[4] or '').strip():q['image_b']=str(row[4]).strip()
        out.append(q)
    return out


def build_cfg(title,format_,theme,aspect,seconds,intro,outro,question_rows,bg_path):
    return {'title':str(title or 'IMPOSSIBLE CHOICES'), 'format':format_,'theme':theme,'aspect':aspect,'fps':24,'seconds_per_question':float(seconds),'intro_seconds':float(intro),'outro_seconds':float(outro),'questions':convert_rows(question_rows), **({'background_image':bg_path} if bg_path else {})}


def make_preview(title,format_,theme,aspect,seconds,intro,outro,rows,bg):
    try:
        cfg=build_cfg(title,format_,theme,aspect,seconds,intro,outro,rows,bg)
        w,h=canvas_dims(aspect,'Preview')
        return round_scene(cfg,w,h,0)
    except Exception as e:raise gr.Error(str(e)) from e


def package_output(directory:Path,cfg:dict,music_file: str|None):
    saved_cfg=json.loads(json.dumps(cfg))
    bundle=directory/'source_assets';bundle.mkdir(exist_ok=True)
    for key in ['background_image']:
        path=saved_cfg.get(key)
        if path and Path(path).is_file():
            dest=bundle/('background'+Path(path).suffix.lower())
            shutil.copyfile(path,dest)
            saved_cfg[key]='source_assets/'+dest.name
    for i,q in enumerate(saved_cfg['questions']):
        for key in ('image_a','image_b'):
            path=q.get(key)
            if path and Path(path).is_file():
                dest=bundle/f'round_{i+1:02d}_{key}{Path(path).suffix.lower()}'
                shutil.copyfile(path,dest)
                q[key]='source_assets/'+dest.name
    (directory/'project_portable.json').write_text(json.dumps(saved_cfg,indent=2,ensure_ascii=False),encoding='utf-8')
    zip_path=directory/'choiceforge_export.zip'
    with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as zf:
        for file in directory.rglob('*'):
            if file.is_file() and file!=zip_path:
                zf.write(file,file.relative_to(directory))
    return str(zip_path)


def run_render(title,format_,theme,aspect,seconds,intro,outro,rows,bg,music_mode,music_file,make_sfx,quality,
               narration_on,voice_choice,rate,pauses,think,energy,
               intro_speech,outro_speech,motion,progress=gr.Progress()):
    try:
        cfg=build_cfg(title,format_,theme,aspect,seconds,intro,outro,rows,bg)
        cfg['motion']=motion
        cfg['narration']={
            'enabled':bool(narration_on),
            'provider':'kokoro',
            'voice':VOICE_PRESETS[voice_choice],
            'rate':int(rate),'pitch':0,'pause':float(pauses),
            'thinking_seconds':float(think),'energy':energy,'auto_extend':True,
            'quality_mode':'natural_only','delivery_mode':'conversational',
            'intro_text':intro_speech.strip(),'outro_text':outro_speech.strip()
        }
        if music_mode == 'Upload my own music' and not music_file:
            raise ValueError('Upload an audio file or choose a different music source')
        directory=OUTPUTS/(time.strftime('%Y%m%d_%H%M%S')+'_'+re.sub('[^a-z0-9]+','-',title.lower())[:26])
        idx=2;original=directory
        while directory.exists():
            directory=Path(str(original)+'_'+str(idx));idx+=1
        progress(0,desc='Preparing scenes')
        def cb(fraction,message):progress(fraction,desc=message)
        result=render(cfg,directory,quality=quality,music_mode={'No background music':'none','Automatically pick from my licensed library':'auto','Upload my own music':'custom'}[music_mode],custom_music=music_file,include_sfx=make_sfx,callback=cb)
        archive=package_output(directory,cfg,music_file)
        music_result='Auto-selected track: '+(Path(result['audio_selected']).name if result['audio_selected'] else 'none in library (add a licensed track first)') if music_mode.startswith('Automatically') else ('Custom music' if music_mode.startswith('Upload') else 'No music')
        msg=f"### Export complete\n**Video:** {result['resolution']} · **Duration:** {result['duration']} sec · **Audio:** {music_result} · **Voice:** {result["voice_segments"]} timed lines.\n\nDownload the ZIP to get the full MP4, no-music MP4, thumbnail, scene timeline, project JSON, original SFX and music license report. Review before uploading."
        return str(result['video']), str(result['clean']), str(result['thumbnail']),archive,msg
    except Exception as e:
        raise gr.Error(str(e)) from e


def preview_voice(voice_choice,rate):
    import tempfile
    try:
        settings={**default_narration(), 'provider':'kokoro',
                  'voice':VOICE_PRESETS[voice_choice], 'rate':int(rate),
                  'quality_mode':'natural_only','delivery_mode':'conversational'}
        tmp=Path(tempfile.mkdtemp(prefix='choiceforge_voice_'))
        dest=tmp/'free_voice_sample.wav'
        voice_clip('Would you rather fly above the clouds... or explore another planet? Think about it!',dest,settings)
        return str(dest)
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def download_free_voice():
    try:
        downloaded=install_models()
        status=model_status()
        if not all(status.values()):raise RuntimeError('Verification failed')
        return ('✅ Free local neural voices ready. ' +
                ('Downloaded: ' + ', '.join(downloaded) if downloaded else 'Already installed.'))
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def add_licensed_track(file_path,title,source,license_type,moods,attribution):
    if not file_path:raise gr.Error('Upload a music file first')
    if not title or not source or not license_type:raise gr.Error('Title, source, and license are required')
    if license_type == 'Creative Commons — attribution required' and not attribution.strip():raise gr.Error('Paste the exact attribution text')
    library=ROOT/'assets'/'music'
    library.mkdir(parents=True,exist_ok=True)
    manifest=library/'tracks.json'
    items=json.loads(manifest.read_text(encoding='utf-8')) if manifest.exists() else []
    filename=re.sub('[^a-z0-9-_]+','-',title.lower()).strip('-')[:55]+Path(file_path).suffix.lower()
    if not filename or filename.startswith('.'):raise gr.Error('Invalid filename')
    if not filename.endswith(('.mp3','.wav','.m4a','.ogg','.flac')):raise gr.Error('Upload an MP3, WAV, M4A, OGG or FLAC file')
    target=library/filename
    shutil.copyfile(file_path,target)
    record={'file':filename,'title':title,'source':source,'license':license_type,'moods':[x.strip().lower() for x in moods.split(',') if x.strip()],'attribution':attribution.strip()}
    items=[r for r in items if r.get('file')!=filename]+[record]
    manifest.write_text(json.dumps(items,indent=2,ensure_ascii=False),encoding='utf-8')
    return f'Added **{title}**. The auto-selector can now match it using your mood tags. Library contains {len(items)} track(s). Your license claims have not been independently verified.'

def trivia_zip_list(text):
    lines=[line.strip() for line in str(text or '').replace(',', '\n').splitlines() if line.strip()]
    return lines or None


def trivia_overview(zip_text, per_type):
    try:
        paths=resolve_zips(trivia_zip_list(zip_text))
        groups=load_bank(paths)
        episodes=build_episodes(groups, int(per_type))
        if not episodes:
            raise ValueError('Not enough questions to build a video with that many from every type')
        return summary(groups, episodes, int(per_type))
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def run_trivia(zip_text, per_type, episode, render_all, think, answer, intro, outro, aspect, quality, theme, progress=gr.Progress()):
    try:
        paths=resolve_zips(trivia_zip_list(zip_text))
        number=int(episode or 1)
        selected=None if render_all else [number]
        def cb(fraction, message):
            progress(min(0.99, max(0.01, float(fraction) if fraction else 0.01)), desc=message)
        result=render_series(paths, OUTPUTS/'visual_trivia', per_type=int(per_type), episodes=selected, workers=1,
                             theme=theme, aspect=aspect, quality=quality, think=float(think), answer=float(answer),
                             intro=float(intro), outro=float(outro), include_sfx=True, callback=cb)
        first=result['episodes'][0]
        status=f"### Trivia export\n{result['summary']}\n\n**This file:** episode {first['episode']:03d} · {first['duration']} sec · {first['resolution']}\n\nFolder: `{result['output']}`"
        return first['video'], first['thumbnail'], first['answers'], status
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


CSS='''
.gradio-container {max-width: 1140px !important; margin: auto !important;}
#masthead {background: linear-gradient(118deg,#10152a,#2a1745);padding:27px 28px;border-radius:18px;color:#f9f7ff;}
#masthead h1 {color:#fff !important;font-size: 2.25rem;}
#masthead p {color:#c7cde7 !important;}
'''

with gr.Blocks(title='ChoiceForge Studio — original interactive videos') as demo:
    gr.Markdown('''# ⚡ ChoiceForge Studio
**Original interactive Shorts and long-form choice videos — made locally with your own assets.**

**100% free local creator:** Kokoro neural narration, original motion graphics, and licensed music imports. No paid APIs or accounts. Download the free voice model once, then generate offline. Music is never scraped from YouTube.''',elem_id='masthead')
    with gr.Tab('🎬 Video Studio'):
        with gr.Row():
            with gr.Column(scale=2):
                title=gr.Textbox(value='IMPOSSIBLE CHOICES',label='Episode title')
                with gr.Row():
                    format_=gr.Dropdown(choices=['Would You Rather','This or That','Pick One'],value='Would You Rather',label='Format')
                    theme=gr.Dropdown(choices=list(THEMES),value='Electric Midnight',label='Visual theme')
                with gr.Row():
                    aspect=gr.Radio(choices=['9:16','16:9'],value='9:16',label='Video layout')
                    quality=gr.Radio(choices=['Preview','Full HD'],value='Preview',label='Export resolution')
                with gr.Row():
                    seconds=gr.Slider(minimum=2,maximum=15,step=.5,value=5,label='Seconds per question')
                    intro=gr.Slider(minimum=.5,maximum=5,step=.5,value=1.5,label='Intro (sec)')
                    outro=gr.Slider(minimum=.5,maximum=5,step=.5,value=1.5,label='Outro (sec)')
                motion=gr.Radio(choices=['Off','Gentle','Dynamic'],value='Gentle',label='Kinetic camera motion (subtle zoom + drift)')
                bg=gr.Image(type='filepath',label='Optional custom background photo (overlaid with gradients)')
            with gr.Column(scale=1):
                gr.Markdown('### How to make a good Short\n1. Use 3–6 genuinely interesting choices.\n2. Keep each option readable in under 2 seconds.\n3. Use 4–6 seconds for each round.\n4. Add a distinctive reason or theme across the episode.\n5. Review before publishing — templated spam is not monetization-safe.')
                gr.Markdown('**Photo panels:** optional per-round local image paths can be added to the two rightmost columns of the question table below. Only use pictures you own or have permission to use.')
        question_table=gr.Dataframe(headers=['Option A','Option B','Creator note','Image A path (optional)','Image B path (optional)'],datatype=['str']*5,value=default_rows(),row_count=(5,'dynamic'),column_count=(5,'fixed'),label='Edit all questions, notes and option imagery',wrap=True,interactive=True)
        with gr.Row():
            preview_button=gr.Button('👁 Preview a round',variant='secondary')
            preview_img=gr.Image(label='Composition preview',height=425)
        with gr.Accordion('🎙️ Natural voice director — speech timing automatically extends scenes',open=True):
            gr.Markdown('**100% free narrator:** Kokoro runs on your laptop, without any account, subscription, API key, or robotic voice. First download the free model (roughly 140 MB). Internet is needed for installation only. American and British English accents are available; emotion is not guaranteed.')
            model_button=gr.Button('⬇ DOWNLOAD FREE VOICE MODEL (one time)',variant='secondary')
            model_status_text=gr.Markdown('Neural model status: install before auditioning or rendering.')
            model_button.click(download_free_voice,[],[model_status_text])
            narration_on=gr.Checkbox(label='Narrate this video automatically',value=True)
            voice_choice=gr.Dropdown(choices=list(VOICE_PRESETS),value='US · Heart (woman)',label='Narrator / accent')
            with gr.Row():
                rate=gr.Slider(-25,25,value=0,step=1,label='Speaking rate adjustment (%)')
                energy=gr.Dropdown(['Warm conversational','Energetic game-show host','Dramatic suspense','Calm storytelling'],value='Warm conversational',label='Pacing profile (subtle; not voice acting)')
            with gr.Row():
                pauses=gr.Slider(0,1.5,value=.45,step=.05,label='Extra pause between clips (legacy projects)')
                think=gr.Slider(.5,8,value=2.0,step=.1,label='Thinking/countdown time after spoken question (sec)')
            with gr.Row():
                intro_speech=gr.Textbox(value=default_narration()['intro_text'],label='Intro voice script',lines=2)
                outro_speech=gr.Textbox(value=default_narration()['outro_text'],label='Outro voice script',lines=2)
            sample_voice=gr.Button('▶ HEAR FREE NEURAL VOICE (preview)',variant='primary')
            sample_voice_audio=gr.Audio(label='Narrator audition',interactive=False)
            sample_voice.click(preview_voice,[voice_choice,rate],[sample_voice_audio])
            gr.Markdown('**Free offline rendering:** no paid APIs are called. **Automatic scene timing:** The app measures actual narrated clips and stretches each round to avoid cutting words or countdown thinking time. Your “seconds per question” slider is a *minimum* while voice is enabled. For short videos, use concise options.')
        with gr.Accordion('🎵 Soundtrack and sound effects',open=True):
            with gr.Row():
                music_mode=gr.Radio(['No background music','Automatically pick from my licensed library','Upload my own music'],value='No background music',label='Music source')
                music_file=gr.Audio(type='filepath',sources=['upload'],label='Optional licensed MP3/WAV (for “Upload my own music”)')
            sfx=gr.Checkbox(label='Include original, generated countdown sound effects',value=True)
            gr.Markdown('**YouTube Audio Library:** open [YouTube Studio → Audio Library](https://www.youtube.com/audiolibrary), filter by mood and attribution, download a permitted track, then upload it here or add it to your library in the next tab. The app does **not** fetch or rip music from YouTube.')
        render_button=gr.Button('🎥 RENDER VIDEO',variant='primary',size='lg')
        status=gr.Markdown()
        with gr.Row():
            video=gr.Video(label='Finished MP4')
            thumb=gr.Image(label='Thumbnail',height=320)
        with gr.Row():
            clean=gr.File(label='Clean video (no music or SFX)')
            archive=gr.File(label='Complete export ZIP (all deliverables)')
        preview_button.click(make_preview,[title,format_,theme,aspect,seconds,intro,outro,question_table,bg],[preview_img])
        render_button.click(run_render,[title,format_,theme,aspect,seconds,intro,outro,question_table,bg,music_mode,music_file,sfx,quality,narration_on,voice_choice,rate,pauses,think,energy,intro_speech,outro_speech,motion],[video,clean,thumb,archive,status])
    with gr.Tab('🧩 Visual trivia'):
        gr.Markdown('''### Picture quizzes from the trivia ZIPs
Each video takes **5 questions from every puzzle type** in the ZIP (count, odd one out, rebus, mirror, and the rest). The picture and the four choices stay up first. **The correct answer and a short explanation appear at the end of that question.**

Put more trivia ZIPs in this folder, or list their paths below, one per line. Questions are used once, in order, until a type runs out.''')
        trivia_zips=gr.Textbox(value=str(ROOT/'visual_trivia_5000_COMPLETE.zip'), lines=2, label='Trivia ZIP files')
        with gr.Row():
            trivia_per=gr.Slider(minimum=1, maximum=8, step=1, value=5, label='Questions from each type')
            trivia_episode=gr.Number(value=1, precision=0, label='Episode number')
            trivia_all=gr.Checkbox(label='Render every episode', value=False)
        with gr.Row():
            trivia_think=gr.Slider(minimum=4, maximum=12, step=0.5, value=7, label='Seconds to think (medium)')
            trivia_answer=gr.Slider(minimum=2, maximum=8, step=0.5, value=4, label='Seconds the answer stays up')
        with gr.Row():
            trivia_intro=gr.Slider(minimum=0.5, maximum=5, step=0.5, value=2, label='Intro (sec)')
            trivia_outro=gr.Slider(minimum=0.5, maximum=5, step=0.5, value=2, label='Outro (sec)')
        with gr.Row():
            trivia_aspect=gr.Radio(choices=['9:16', '16:9'], value='9:16', label='Video layout')
            trivia_quality=gr.Radio(choices=['Preview', 'Full HD'], value='Preview', label='Export resolution')
            trivia_theme=gr.Dropdown(choices=list(THEMES), value='Electric Midnight', label='Visual theme')
        with gr.Row():
            trivia_scan=gr.Button('Scan ZIPs', variant='secondary')
            trivia_render=gr.Button('Render trivia video', variant='primary')
        trivia_plan=gr.Markdown()
        trivia_status=gr.Markdown()
        with gr.Row():
            trivia_video_out=gr.Video(label='Trivia MP4')
            trivia_thumb=gr.Image(label='Thumbnail', height=320)
        trivia_answers=gr.File(label='Answer key')
        trivia_scan.click(trivia_overview, [trivia_zips, trivia_per], [trivia_plan])
        trivia_render.click(run_trivia, [trivia_zips, trivia_per, trivia_episode, trivia_all, trivia_think, trivia_answer, trivia_intro, trivia_outro, trivia_aspect, trivia_quality, trivia_theme], [trivia_video_out, trivia_thumb, trivia_answers, trivia_status])
    with gr.Tab('💡 Free idea bank'):
        gr.Markdown('### Offline question bank — no account needed\nGenerate original combinations to get started, then edit them so each episode has its own personality.')
        with gr.Row():
            category=gr.Dropdown(list(BANK),value='Fantasy',label='Topic')
            how_many=gr.Slider(minimum=2,maximum=8,step=1,value=5,label='Number of questions')
            seed=gr.Number(label='Variation seed (change for new ideas)',value=42,precision=0)
            offline_button=gr.Button('Generate offline ideas')
        offline_button.click(rows_for,[category,how_many,seed],[question_table])
        gr.Markdown('All idea generation here is from the free built-in question bank. No paid API is needed.')
        gr.Markdown('Return to **Video Studio** to preview, edit and render the generated questions.')
    with gr.Tab('🎶 My licensed music library'):
        gr.Markdown('### Upload tracks you have permission to use\nThe auto-selector searches only your local, documented library. It matches mood tags (for example `upbeat, pop, electronic`) to the visual theme. **License and attribution are your responsibility.**')
        track_file=gr.Audio(sources=['upload'],type='filepath',label='Add licensed music file')
        with gr.Row():
            track_title=gr.Textbox(label='Track title',placeholder='Neon Energy')
            track_source=gr.Textbox(label='Where did you get this track?',placeholder='YouTube Audio Library (downloaded from Studio)')
        with gr.Row():
            track_license=gr.Dropdown(['YouTube Audio Library — attribution not required','Creative Commons — attribution required','Purchased commercial license','Original music I created'],label='License evidence / category')
            track_moods=gr.Textbox(label='Moods (comma separated)',value='energetic, upbeat, electronic')
        track_attr=gr.Textbox(label='Attribution text — paste exact required credit, if applicable',lines=2)
        add_btn=gr.Button('Save track in my library',variant='primary')
        track_out=gr.Markdown()
        add_btn.click(add_licensed_track,[track_file,track_title,track_source,track_license,track_moods,track_attr],[track_out])
    with gr.Tab('📚 Setup & rights'):
        gr.Markdown('''### Getting started

1. Install Python 3.11+ and FFmpeg on your computer.
2. Run `pip install -r requirements.txt` and `python app.py`.
3. Visit the local browser address shown in the terminal.
4. Edit the question table, choose a visual theme, and preview a round.
5. Render **Preview** first. Switch to **Full HD** for the version you actually publish.
6. Check every claim, visual and audio license; upload the MP4 yourself.

### What's included
- Output files: `video_final.mp4`, `video_clean.mp4`, `thumbnail.png`, `project_portable.json`, `timeline.csv`, `description.txt`, `MUSIC_LICENSE.txt` , `narration.wav` (if enabled), `narration_cues.json`, `narration_script.txt`, and generated `original_sfx.wav`.
- **Both portrait (Shorts) and landscape (16:9)**, with 24fps, H.264/AAC encoding and separate clean export.
- Add licensed background music manually, or let the app pick from tracks **you** imported with license metadata.
- User-supplied custom background images and optional option-card photos.

### What this version deliberately does not do
It does **not** scrape YouTube music, verify music licenses, automatically generate photos, guarantee monetization, or upload without review. Kokoro narration runs offline after the free model download. The source project JSON and clean video make further editing in a video editor easy. Full HD rendering can take several minutes on slower laptops.

For editable source code and CLI batch rendering, see `README.md`.''')

if __name__=='__main__':
    demo.launch(server_name='127.0.0.1',server_port=int(os.environ.get('PORT',7860)),share=False,show_error=True,css=CSS)
