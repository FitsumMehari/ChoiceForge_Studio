import json
import shutil
import subprocess
import zipfile
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from trivia_video import (
    build_episodes, load_bank, mix_sequence, render_episode, resolve_zips, trivia_frame,
)

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / 'visual_trivia_5000_COMPLETE.zip'


def _jpeg():
    image = Image.new('RGB', (80, 80), (220, 230, 245))
    buffer = BytesIO()
    image.save(buffer, format='JPEG')
    return buffer.getvalue()


def _question(qid, category, difficulty, letter='A'):
    return {
        'id': qid,
        'category': category,
        'difficulty': difficulty,
        'image': f'images/{qid}.jpg',
        'question': f'Question {qid}?',
        'options': {'A': 'One', 'B': 'Two', 'C': 'Three', 'D': 'Four'},
        'correct_answer': letter,
        'explanation': f'{qid} is {letter}.',
        'image_bytes': 10,
    }


def _write_zip(path: Path, questions: list[dict]):
    jpeg = _jpeg()
    with zipfile.ZipFile(path, 'w') as archive:
        for question in questions:
            archive.writestr(f"json/{question['id']}.json", json.dumps(question))
            archive.writestr(question['image'], jpeg)


def test_mix_alternates_the_remainder():
    assert mix_sequence(5, 0) == ['easy', 'easy', 'medium', 'medium', 'difficult']
    assert mix_sequence(5, 1) == ['easy', 'medium', 'medium', 'difficult', 'difficult']


def test_episodes_take_five_from_each_type_without_repeats():
    groups = {}
    for category, total in (('count_animals', 12), ('number_hunt', 10)):
        groups[category] = []
        for index in range(total):
            difficulty = ('easy', 'medium', 'difficult')[index % 3]
            groups[category].append(_question(f'{category[:3].upper()}_{index:02d}', category, difficulty))
    episodes = build_episodes(groups, 2)
    assert len(episodes) == 5
    seen = []
    for episode in episodes:
        assert len(episode) == 4
        counts = {}
        for question in episode:
            counts[question['category']] = counts.get(question['category'], 0) + 1
            seen.append(question['id'])
        assert counts == {'count_animals': 2, 'number_hunt': 2}
        assert [question['category'] for question in episode] == ['count_animals', 'number_hunt', 'count_animals', 'number_hunt']
    assert len(seen) == len(set(seen))


def test_real_bank_builds_balanced_episodes():
    if not BANK.is_file():
        pytest.skip('visual trivia ZIP is not in the project folder')
    groups = load_bank([BANK])
    episodes = build_episodes(groups, 5)
    assert len(episodes) == min(len(items) for items in groups.values()) // 5
    assert len(episodes) == 50
    seen = set()
    for episode in episodes:
        assert len(episode) == 50
        counts = {}
        for question in episode:
            counts[question['category']] = counts.get(question['category'], 0) + 1
            assert question['id'] not in seen
            seen.add(question['id'])
        assert set(counts) == set(groups)
        assert set(counts.values()) == {5}
    animals = [question for question in episodes[0] if question['category'] == 'count_animals']
    assert [question['difficulty'] for question in animals] == ['easy', 'easy', 'medium', 'medium', 'difficult']
    assert animals[0]['id'] == 'VIS_00001'


def test_answer_frame_differs_and_matches_canvas():
    question = _question('VIS_1', 'count_animals', 'easy', 'B')
    question['options']['B'] = '2'
    question['explanation'] = 'There are 2 cats in the picture.'
    photo = Image.new('RGB', (64, 80), (180, 200, 230))
    ask = trivia_frame(question, photo, 540, 960, 1, 5, False, 'Electric Midnight')
    reveal = trivia_frame(question, photo, 540, 960, 1, 5, True, 'Electric Midnight')
    assert ask.size == (540, 960)
    assert reveal.size == ask.size
    assert ask.tobytes() != reveal.tobytes()


@pytest.mark.skipif(not shutil.which('ffmpeg'), reason='Requires FFmpeg')
def test_render_episode_shows_answer_segment(tmp_path):
    questions = [_question('VIS_1', 'count_animals', 'easy', 'B'), _question('VIS_2', 'number_hunt', 'difficult', 'C')]
    archive = tmp_path / 'pack.zip'
    _write_zip(archive, questions)
    groups = load_bank(resolve_zips([archive]))
    episode = build_episodes(groups, 1)[0]
    result = render_episode(episode, tmp_path / 'out', think=4, answer=2, intro=0.5, outro=0.5, include_sfx=True)
    video = Path(result['video'])
    assert video.is_file()
    assert Path(result['answers']).read_text(encoding='utf-8').find('B. Two') != -1
    probe = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration:stream=width,height,codec_name', '-of', 'json', str(video)],
        capture_output=True, text=True, check=True,
    )
    payload = json.loads(probe.stdout)
    assert any(stream.get('width') == 540 and stream.get('height') == 960 and stream.get('codec_name') == 'h264' for stream in payload['streams'])
    assert any(stream.get('codec_name') == 'aac' for stream in payload['streams'])
    duration = float(payload['format']['duration'])
    # easy 4-1.5=2.5 clamped to 3, plus difficult 4+2=6, answers 2+2, intro/outro 0.5+0.5
    assert duration == pytest.approx(3 + 2 + 6 + 2 + 0.5 + 0.5, abs=0.4)
