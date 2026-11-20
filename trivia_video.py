"""Visual trivia videos from the question ZIPs.

Each episode takes the same number of questions from every puzzle type.
The picture and the four choices play first. The correct answer and the
explanation appear at the end of that question.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import wave
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from renderer import THEMES, bg_image, canvas_dims, font

ROOT = Path(__file__).resolve().parent

CATEGORY_ORDER = (
    'count_animals',
    'number_hunt',
    'odd_one_out',
    'rebus_phrase',
    'complete_pattern',
    'count_shapes',
    'spot_differences',
    'mirror_image',
    'picture_word',
    'visual_math',
)

CATEGORY_LABELS = {
    'count_animals': 'Count Animals',
    'number_hunt': 'Number Hunt',
    'odd_one_out': 'Odd One Out',
    'rebus_phrase': 'Rebus',
    'complete_pattern': 'Patterns',
    'count_shapes': 'Count Shapes',
    'spot_differences': 'Differences',
    'mirror_image': 'Mirror',
    'picture_word': 'Picture Word',
    'visual_math': 'Visual Math',
}

DIFFICULTIES = ('easy', 'medium', 'difficult')
_BANK_CACHE: dict = {}


def category_label(category: str) -> str:
    if category in CATEGORY_LABELS:
        return CATEGORY_LABELS[category]
    return category.replace('_', ' ').title()


def discover_zips(folder: Path = ROOT) -> list[Path]:
    """ZIP files in a folder that contain trivia JSON questions."""
    found = []
    if not folder.is_dir():
        return found
    for path in sorted(folder.glob('*.zip')):
        try:
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
        except zipfile.BadZipFile:
            continue
        if any(name.startswith('json/') and name.endswith('.json') for name in names):
            found.append(path)
    return found


def resolve_zips(raw_paths: list[str | Path] | None) -> list[Path]:
    if not raw_paths:
        found = discover_zips(ROOT)
        if not found:
            raise FileNotFoundError('No trivia ZIP files found in the project folder')
        return found
    found: list[Path] = []
    for raw in raw_paths:
        text = str(raw).strip()
        if not text:
            continue
        path = Path(text).expanduser()
        if path.is_dir():
            found.extend(discover_zips(path))
        elif path.is_file():
            found.append(path.resolve())
        else:
            raise FileNotFoundError(f'ZIP not found: {path}')
    unique = []
    seen = set()
    for path in found:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(path.resolve())
    if not unique:
        raise FileNotFoundError('No trivia ZIP files found')
    return unique


def load_bank(zip_paths: list[Path]) -> dict[str, list[dict]]:
    """Questions grouped by category. Results are cached by ZIP timestamp."""
    resolved = [Path(path).resolve() for path in zip_paths]
    key = tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in resolved)
    cached = _BANK_CACHE.get(key)
    if cached is not None:
        return cached
    groups: dict[str, list[dict]] = {}
    seen_ids = set()
    for path in resolved:
        with zipfile.ZipFile(path) as archive:
            names = sorted(name for name in archive.namelist() if name.startswith('json/') and name.endswith('.json'))
            if not names:
                raise ValueError(f'{path.name} has no json/ question files')
            for name in names:
                question = json.loads(archive.read(name))
                qid = str(question.get('id') or '')
                if not qid or qid in seen_ids:
                    continue
                _require_question(question, name)
                question['zip_path'] = str(path)
                groups.setdefault(question['category'], []).append(question)
                seen_ids.add(qid)
    if not groups:
        raise ValueError('The ZIP files have no questions')
    for items in groups.values():
        items.sort(key=lambda question: question['id'])
    _BANK_CACHE[key] = groups
    return groups


def _require_question(question: dict, name: str):
    required = ('id', 'category', 'difficulty', 'image', 'question', 'options', 'correct_answer', 'explanation')
    missing = [field for field in required if not question.get(field)]
    if missing:
        raise ValueError(f'{name} is missing {", ".join(missing)}')
    if question['difficulty'] not in DIFFICULTIES:
        raise ValueError(f"{question['id']} has an unknown difficulty")
    options = question['options']
    if not isinstance(options, dict) or any(letter not in options for letter in 'ABCD'):
        raise ValueError(f"{question['id']} must have options A, B, C and D")
    if question['correct_answer'] not in options:
        raise ValueError(f"{question['id']} correct answer is not one of the options")


def mix_sequence(per_type: int, episode_index: int) -> list[str]:
    """Difficulty order for one type inside one episode.

    The bank is 30% easy, 40% medium, 30% difficult. Even episodes spend the
    remainder on easier questions. Odd episodes spend it on harder ones.
    """
    if per_type < 1:
        raise ValueError('Questions per type must be at least 1')
    counts = {
        'easy': per_type * 3 // 10,
        'medium': per_type * 4 // 10,
        'difficult': per_type * 3 // 10,
    }
    leftover = per_type - sum(counts.values())
    order = ('easy', 'medium', 'difficult') if episode_index % 2 == 0 else ('difficult', 'medium', 'easy')
    slot = 0
    while leftover:
        counts[order[slot % 3]] += 1
        leftover -= 1
        slot += 1
    return (['easy'] * counts['easy']
            + ['medium'] * counts['medium']
            + ['difficult'] * counts['difficult'])


def _can_fill(bucket: dict[str, list], sequence: list[str]) -> bool:
    need = Counter(sequence)
    available = {name: len(bucket.get(name, ())) for name in DIFFICULTIES}
    missing = 0
    surplus = 0
    for name in DIFFICULTIES:
        gap = available[name] - need[name]
        if gap >= 0:
            surplus += gap
        else:
            missing -= gap
    return missing <= surplus and sum(available.values()) >= len(sequence)


def _take(bucket: dict[str, list], sequence: list[str]) -> list[dict]:
    picked = []
    for difficulty in sequence:
        if bucket[difficulty]:
            picked.append(bucket[difficulty].pop(0))
            continue
        donor = max(DIFFICULTIES, key=lambda name: len(bucket[name]))
        if not bucket[donor]:
            raise RuntimeError('Ran out of questions while building an episode')
        picked.append(bucket[donor].pop(0))
    return picked


def present_categories(groups: dict[str, list[dict]]) -> list[str]:
    ordered = [name for name in CATEGORY_ORDER if groups.get(name)]
    extras = sorted(name for name in groups if name not in CATEGORY_ORDER and groups[name])
    return ordered + extras


def build_episodes(groups: dict[str, list[dict]], per_type: int = 5) -> list[list[dict]]:
    """Full episodes only. A video is skipped once any puzzle type runs out."""
    if not 1 <= int(per_type) <= 20:
        raise ValueError('Questions per type must be from 1 to 20')
    per_type = int(per_type)
    categories = present_categories(groups)
    if not categories:
        return []
    buckets = {
        category: {difficulty: [question for question in groups[category] if question['difficulty'] == difficulty]
                   for difficulty in DIFFICULTIES}
        for category in categories
    }
    episodes = []
    index = 0
    while all(_can_fill(buckets[category], mix_sequence(per_type, index)) for category in categories):
        sequence = mix_sequence(per_type, index)
        picked = {category: _take(buckets[category], sequence) for category in categories}
        episode = [picked[category][slot] for slot in range(per_type) for category in categories]
        episodes.append(episode)
        index += 1
    return episodes


def think_seconds(difficulty: str, base: float) -> float:
    delta = {'easy': -1.5, 'medium': 0.0, 'difficult': 2.0}[difficulty]
    return max(3.0, float(base) + delta)


def summary(groups: dict[str, list[dict]], episodes: list[list[dict]], per_type: int) -> str:
    categories = present_categories(groups)
    used = len(episodes) * per_type
    lines = [
        f'{len(episodes)} video{"s" if len(episodes) != 1 else ""}',
        f'{per_type} questions from each of {len(categories)} puzzle types',
        f'{used * len(categories)} questions placed, answers at the end of each question',
    ]
    for category in categories:
        lines.append(f'{category_label(category)}: {len(groups[category])} in the ZIP, {used} used')
    return '\n'.join(lines)


def _wrapped(text: str, max_width: int, start: int, minimum: int, max_lines: int, bold=False, heavy=False):
    text = ' '.join(str(text).split()) or ' '
    probe = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    size = max(minimum, start)
    while size >= minimum:
        face = font(size, bold=bold, heavy=heavy)
        lines: list[str] = []
        current = ''
        for word in text.split():
            candidate = (current + ' ' + word).strip()
            width = probe.textbbox((0, 0), candidate, font=face)[2]
            if width <= max_width or not current:
                current = candidate
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        if len(lines) <= max_lines:
            return lines, face
        size -= 1
    face = font(minimum, bold=bold, heavy=heavy)
    return lines[:max_lines], face


def _draw_center(draw, box, lines, face, fill, gap=0.18):
    x0, y0, x1, y1 = box
    line_h = int(face.size * (1 + gap))
    block = line_h * max(1, len(lines))
    top = y0 + max(0, (y1 - y0 - block) / 2)
    for index, line in enumerate(lines):
        draw.text(((x0 + x1) / 2, top + index * line_h + face.size / 2), line, font=face, fill=fill, anchor='mm')


def layout_boxes(w: int, h: int) -> dict:
    margin = max(16, int(min(w, h) * 0.035))
    portrait = w < h
    if portrait:
        return {
            'portrait': True,
            'header': (margin, int(h * 0.028), w - margin, int(h * 0.088)),
            'image': (margin, int(h * 0.105), w - margin, int(h * 0.535)),
            'question': (margin, int(h * 0.548), w - margin, int(h * 0.640)),
            'options': (margin, int(h * 0.652), w - margin, int(h * 0.838)),
            'footer': (margin, int(h * 0.852), w - margin, int(h * 0.972)),
        }
    return {
        'portrait': False,
        'header': (margin, int(h * 0.035), w - margin, int(h * 0.125)),
        'image': (margin, int(h * 0.155), int(w * 0.50), int(h * 0.76)),
        'question': (int(w * 0.525), int(h * 0.155), w - margin, int(h * 0.31)),
        'options': (int(w * 0.525), int(h * 0.33), w - margin, int(h * 0.76)),
        'footer': (margin, int(h * 0.79), w - margin, int(h * 0.96)),
    }


def _blit_photo(base: Image.Image, photo: Image.Image, box, radius: int):
    x0, y0, x1, y1 = box
    width, height = x1 - x0, y1 - y0
    if width < 8 or height < 8:
        return
    photo = photo.convert('RGB')
    scale = min(width / photo.width, height / photo.height)
    resized = photo.resize((max(1, int(round(photo.width * scale))), max(1, int(round(photo.height * scale)))), Image.Resampling.LANCZOS)
    card = Image.new('RGBA', (width, height), (255, 255, 255, 255))
    card.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    mask = Image.new('L', (width, height), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, width - 1, height - 1), radius=min(radius, width // 2, height // 2), fill=255)
    base.paste(card, (x0, y0), mask)


def _option_grid(area) -> list[tuple]:
    x0, y0, x1, y1 = area
    gap = max(8, int(min(x1 - x0, y1 - y0) * 0.045))
    cell_w = (x1 - x0 - gap) // 2
    cell_h = (y1 - y0 - gap) // 2
    boxes = []
    for index in range(4):
        column, row = index % 2, index // 2
        left = x0 + column * (cell_w + gap)
        top = y0 + row * (cell_h + gap)
        boxes.append((left, top, left + cell_w, top + cell_h))
    return boxes


def _draw_option(draw, box, letter: str, text: str, state: str, short: int):
    x0, y0, x1, y1 = box
    if state == 'correct':
        fill, outline = (14, 122, 82, 245), (186, 255, 214, 255)
        fg = (255, 255, 255, 255)
        badge_fill, badge_fg = (255, 255, 255, 255), (10, 78, 52, 255)
    elif state == 'wrong':
        fill, outline = (255, 255, 255, 20), (255, 255, 255, 40)
        fg = (214, 220, 232, 145)
        badge_fill, badge_fg = (255, 255, 255, 36), (226, 230, 240, 160)
    else:
        fill, outline = (255, 255, 255, 40), (255, 255, 255, 185)
        fg = (255, 255, 255, 255)
        badge_fill, badge_fg = (255, 255, 255, 235), (16, 20, 42, 255)
    radius = max(12, int(min(x1 - x0, y1 - y0) * 0.18))
    draw.rounded_rectangle((x0, y0, x1, y1), radius=radius, fill=fill, outline=outline, width=max(2, short // 170))
    side = min(y1 - y0 - 12, int((x1 - x0) * 0.24), int(short * 0.075))
    side = max(22, side)
    badge_x = x0 + max(8, int((y1 - y0 - side) * 0.35))
    badge_y = y0 + (y1 - y0 - side) // 2
    draw.rounded_rectangle((badge_x, badge_y, badge_x + side, badge_y + side), radius=side // 2, fill=badge_fill)
    draw.text((badge_x + side / 2, badge_y + side / 2), letter, font=font(max(12, int(side * 0.46)), heavy=True), fill=badge_fg, anchor='mm')
    text_box = (badge_x + side + 8, y0 + 4, x1 - 10, y1 - 4)
    lines, face = _wrapped(text, max(20, text_box[2] - text_box[0]), max(14, int(short * 0.031)), 11, 2, bold=True)
    _draw_center(draw, text_box, lines, face, fg, gap=0.12)


def _draw_footer(draw, box, question: dict, reveal: bool, short: int):
    x0, y0, x1, y1 = box
    radius = max(16, int(short * 0.03))
    if not reveal:
        draw.rounded_rectangle((x0, y0, x1, y1), radius=radius, fill=(255, 255, 255, 28), outline=(255, 255, 255, 90), width=2)
        lines, face = _wrapped('ANSWER AT THE END', x1 - x0 - 24, max(16, int(short * 0.034)), 12, 2, bold=True)
        _draw_center(draw, (x0 + 12, y0 + 6, x1 - 12, y1 - 6), lines, face, (232, 238, 255, 235))
        return
    draw.rounded_rectangle((x0, y0, x1, y1), radius=radius, fill=(10, 108, 74, 245), outline=(176, 255, 210, 255), width=3)
    letter = str(question['correct_answer'])
    answer = str(question['options'][letter])
    side = min(y1 - y0 - 20, int(short * 0.11))
    side = max(36, side)
    badge_x = x0 + 16
    badge_y = y0 + (y1 - y0 - side) // 2
    draw.rounded_rectangle((badge_x, badge_y, badge_x + side, badge_y + side), radius=side // 2, fill=(255, 255, 255, 255))
    draw.text((badge_x + side / 2, badge_y + side / 2), letter, font=font(max(16, int(side * 0.48)), heavy=True), fill=(8, 78, 52, 255), anchor='mm')
    text_left = badge_x + side + 16
    answer_box = (text_left, y0 + 8, x1 - 16, y0 + int((y1 - y0) * 0.48))
    explain_box = (text_left, y0 + int((y1 - y0) * 0.46), x1 - 16, y1 - 8)
    answer_lines, answer_face = _wrapped(answer, answer_box[2] - answer_box[0], max(16, int(short * 0.038)), 12, 2, heavy=True)
    explain_lines, explain_face = _wrapped(question.get('explanation') or '', explain_box[2] - explain_box[0], max(13, int(short * 0.026)), 11, 2, bold=False)
    _draw_center(draw, answer_box, answer_lines, answer_face, (255, 255, 255, 255), gap=0.08)
    _draw_center(draw, explain_box, explain_lines, explain_face, (226, 255, 240, 235), gap=0.08)


_BG_CACHE: dict = {}


def _background(w: int, h: int, theme: str, seed: int) -> Image.Image:
    key = (w, h, theme, seed)
    cached = _BG_CACHE.get(key)
    if cached is None:
        cached = bg_image(w, h, theme, seed=seed)
        _BG_CACHE[key] = cached
    return cached.copy()


def trivia_frame(question: dict, photo: Image.Image, w: int, h: int, index: int, total: int, reveal: bool, theme: str) -> Image.Image:
    base = _background(w, h, theme, 90)
    boxes = layout_boxes(w, h)
    _blit_photo(base, photo, boxes['image'], radius=max(16, int(min(w, h) * 0.028)))
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, 'RGBA')
    short = min(w, h)
    header = f'Q {index} / {total}    ·    {category_label(question["category"]).upper()}    ·    {question["difficulty"].upper()}'
    header_lines, header_face = _wrapped(header, boxes['header'][2] - boxes['header'][0], max(14, int(short * 0.028)), 11, 2, bold=True)
    _draw_center(draw, boxes['header'], header_lines, header_face, (236, 242, 255, 245))
    q_lines, q_face = _wrapped(question['question'], boxes['question'][2] - boxes['question'][0] - 8, max(18, int(short * 0.042)), 13, 3, heavy=True)
    _draw_center(draw, boxes['question'], q_lines, q_face, (255, 255, 255, 255))
    letters = [letter for letter in 'ABCD' if letter in question['options']]
    for letter, box in zip(letters, _option_grid(boxes['options'])):
        if reveal and letter == question['correct_answer']:
            state = 'correct'
        elif reveal:
            state = 'wrong'
        else:
            state = 'open'
        _draw_option(draw, box, letter, str(question['options'][letter]), state, short)
    _draw_footer(draw, boxes['footer'], question, reveal, short)
    base.alpha_composite(overlay)
    return base.convert('RGB')


def title_frame(w: int, h: int, theme: str, kicker: str, title: str, subtitle: str, foot: str, seed: int) -> Image.Image:
    base = _background(w, h, theme, seed)
    draw = ImageDraw.Draw(base, 'RGBA')
    short = min(w, h)
    accent = THEMES[theme]['left']
    kick_lines, kick_face = _wrapped(kicker, int(w * 0.84), max(16, int(short * 0.03)), 12, 2, bold=True)
    _draw_center(draw, (int(w * 0.08), int(h * 0.18), int(w * 0.92), int(h * 0.28)), kick_lines, kick_face, (*accent, 255))
    title_lines, title_face = _wrapped(title, int(w * 0.86), max(32, int(short * 0.078)), 20, 4, heavy=True)
    _draw_center(draw, (int(w * 0.07), int(h * 0.30), int(w * 0.93), int(h * 0.56)), title_lines, title_face, (255, 255, 255, 255), gap=0.12)
    sub_lines, sub_face = _wrapped(subtitle, int(w * 0.8), max(15, int(short * 0.03)), 12, 3, bold=True)
    _draw_center(draw, (int(w * 0.1), int(h * 0.58), int(w * 0.9), int(h * 0.70)), sub_lines, sub_face, (226, 232, 250, 240))
    foot_box = (int(w * 0.1), int(h * 0.76), int(w * 0.9), int(h * 0.88))
    draw.rounded_rectangle(foot_box, radius=max(16, int(short * 0.03)), fill=(*accent, 255))
    foot_lines, foot_face = _wrapped(foot, foot_box[2] - foot_box[0] - 24, max(14, int(short * 0.028)), 11, 2, bold=True)
    _draw_center(draw, foot_box, foot_lines, foot_face, (14, 18, 36, 255))
    return base.convert('RGB')


def _open_photo(archive_cache: dict[str, zipfile.ZipFile], question: dict) -> Image.Image:
    path = question['zip_path']
    archive = archive_cache.get(path)
    if archive is None:
        archive = zipfile.ZipFile(path)
        archive_cache[path] = archive
    with archive.open(question['image']) as handle:
        return Image.open(BytesIO(handle.read())).convert('RGB')


def _write_chimes(path: Path, marks: list[float], total: float, sample_rate=44100):
    count = int((total + 0.05) * sample_rate)
    audio = np.zeros(count, dtype=np.float32)

    def tone(start_s, duration, freq, amp):
        start = int(start_s * sample_rate)
        samples = int(duration * sample_rate)
        if samples <= 0 or start >= count:
            return
        samples = min(samples, count - start)
        axis = np.arange(samples, dtype=np.float32) / sample_rate
        envelope = np.minimum(1, axis / 0.012) * np.maximum(0, 1 - axis / duration) ** 2
        audio[start:start + samples] += np.sin(2 * np.pi * freq * axis) * amp * envelope

    for moment in marks:
        tone(moment, 0.22, 880, 0.11)
        tone(moment + 0.02, 0.30, 1318, 0.055)
    audio = np.clip(audio, -0.95, 0.95)
    with wave.open(str(path), 'wb') as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes((audio * 32767).astype('<i2').tobytes())


def _concat_script(segments: list[tuple[Path, float]]) -> str:
    def escaped(path: Path) -> str:
        return str(path.resolve()).replace('\\', '/').replace("'", r"'\''")

    lines = ['ffconcat version 1.0']
    for path, duration in segments:
        lines.append(f"file '{escaped(path)}'")
        lines.append(f'duration {duration:.3f}')
    lines.append(f"file '{escaped(segments[-1][0])}'")
    return '\n'.join(lines) + '\n'


def render_episode(episode: list[dict], output: str | Path, *, episode_number: int = 1, episode_count: int = 1,
                    theme: str = 'Electric Midnight', aspect: str = '9:16', quality: str = 'Preview',
                    think: float = 7.0, answer: float = 4.0, intro: float = 2.0, outro: float = 2.2,
                    include_sfx: bool = True, fps: int = 24, callback=None) -> dict:
    """Render one trivia episode. Questions already include zip_path and image."""
    if theme not in THEMES:
        raise ValueError(f"Theme must be one of: {', '.join(THEMES)}")
    if aspect not in ('9:16', '16:9'):
        raise ValueError('Aspect must be 9:16 or 16:9')
    if quality not in ('Preview', 'Full HD'):
        raise ValueError('Quality must be Preview or Full HD')
    if not episode:
        raise ValueError('Episode has no questions')
    if not 0.4 <= float(intro) <= 12 or not 0.4 <= float(outro) <= 12:
        raise ValueError('Intro and outro must be between 0.4 and 12 seconds')
    if not 2 <= float(answer) <= 15:
        raise ValueError('Answer time must be between 2 and 15 seconds')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frames = output / '_frames'
    frames.mkdir(exist_ok=True)
    w, h = canvas_dims(aspect, quality)
    categories = list(dict.fromkeys(question['category'] for question in episode))
    per_type = Counter(question['category'] for question in episode)
    title = f'Visual Trivia {episode_number:03d}'
    intro_image = title_frame(
        w, h, theme,
        f'EPISODE  {episode_number:02d}',
        'VISUAL TRIVIA',
        f'{len(episode)} questions  ·  {len(categories)} puzzle types',
        'THE ANSWER SHOWS AT THE END OF EACH QUESTION',
        seed=11,
    )
    outro_image = title_frame(
        w, h, theme,
        'THAT IS THE SET',
        'HOW MANY DID YOU GET?',
        f'{per_type.most_common(1)[0][1]} of every puzzle type',
        'COMMENT YOUR SCORE',
        seed=29,
    )
    archives: dict[str, zipfile.ZipFile] = {}
    segments: list[tuple[Path, float]] = []
    timeline = []
    clock = 0.0
    intro_path = frames / 'intro.png'
    intro_image.save(intro_path)
    intro_image.save(output / 'thumbnail.png')
    segments.append((intro_path, float(intro)))
    timeline.append(('INTRO', clock, clock + float(intro), title))
    clock += float(intro)
    chime_marks = []
    try:
        for index, question in enumerate(episode, 1):
            if callback:
                callback((index - 1) / max(1, len(episode)) * 0.55, f'Drawing question {index} of {len(episode)}')
            photo = _open_photo(archives, question)
            ask = trivia_frame(question, photo, w, h, index, len(episode), False, theme)
            reveal = trivia_frame(question, photo, w, h, index, len(episode), True, theme)
            ask_path = frames / f'q_{index:03d}.png'
            answer_path = frames / f'a_{index:03d}.png'
            ask.save(ask_path)
            reveal.save(answer_path)
            ask_for = think_seconds(question['difficulty'], think)
            answer_for = float(answer)
            segments.append((ask_path, ask_for))
            segments.append((answer_path, answer_for))
            timeline.append(('QUESTION', clock, clock + ask_for, f"{question['id']}  {question['question']}"))
            clock += ask_for
            chime_marks.append(clock)
            letter = question['correct_answer']
            timeline.append(('ANSWER', clock, clock + answer_for, f"{question['id']}  {letter}. {question['options'][letter]}  {question['explanation']}"))
            clock += answer_for
        outro_path = frames / 'outro.png'
        outro_image.save(outro_path)
        segments.append((outro_path, float(outro)))
        timeline.append(('OUTRO', clock, clock + float(outro), 'How many did you get?'))
        clock += float(outro)
        (frames / 'list.ffconcat').write_text(_concat_script(segments), encoding='utf-8')
        sfx_path = output / 'original_sfx.wav'
        if include_sfx:
            _write_chimes(sfx_path, chime_marks, clock)
        if callback:
            callback(0.62, 'Encoding video')
        final = output / 'video_final.mp4'
        _encode(frames / 'list.ffconcat', sfx_path if include_sfx else None, final, fps, clock)
        shutil.rmtree(frames, ignore_errors=True)
    finally:
        for archive in archives.values():
            archive.close()
    _write_sidecars(output, episode, episode_number, episode_count, timeline, title, theme, aspect, quality)
    if callback:
        callback(1.0, 'Done')
    return {
        'video': str(final),
        'thumbnail': str(output / 'thumbnail.png'),
        'answers': str(output / 'answers.txt'),
        'questions': str(output / 'questions.json'),
        'timeline': str(output / 'timeline.csv'),
        'duration': round(clock, 2),
        'resolution': f'{w}x{h}',
        'questions_count': len(episode),
        'episode': episode_number,
    }


def _encode(script: Path, sfx: Path | None, final: Path, fps: int, duration: float):
    import subprocess
    if not shutil.which('ffmpeg'):
        raise RuntimeError('FFmpeg not found on PATH. Install FFmpeg first.')
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', str(script)]
    if sfx is not None:
        command += ['-i', str(sfx)]
    command += ['-vf', f'fps={fps},format=yuv420p', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '19', '-threads', '2']
    if sfx is not None:
        command += ['-c:a', 'aac', '-b:a', '192k']
    else:
        command += ['-an']
    command += ['-t', f'{duration:.3f}', '-movflags', '+faststart', str(final)]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0 or not final.is_file():
        raise RuntimeError('FFmpeg failed: ' + (result.stderr or '')[-1600:])


def _write_sidecars(output: Path, episode, episode_number, episode_count, timeline, title, theme, aspect, quality):
    rows = []
    for kind, start, end, text in timeline:
        rows.append((kind, f'{start:.2f}', f'{end:.2f}', text))
    with (output / 'timeline.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['type', 'start_seconds', 'end_seconds', 'text'])
        writer.writerows(rows)
    answer_lines = [f'{title}', '']
    public = [title, '', 'Visual trivia. Five puzzles from every type. The answer appears at the end of each question.', '']
    saved = []
    for index, question in enumerate(episode, 1):
        letter = question['correct_answer']
        answer_lines.append(f"{index:02d}. [{category_label(question['category'])} · {question['difficulty']}] {question['question']}")
        answer_lines.append(f"    {letter}. {question['options'][letter]}")
        answer_lines.append(f"    {question['explanation']}")
        public.append(f"{index:02d}. {question['question']}")
        saved.append({key: question[key] for key in ('id', 'category', 'difficulty', 'question', 'options', 'correct_answer', 'explanation', 'image')})
    (output / 'answers.txt').write_text('\n'.join(answer_lines) + '\n', encoding='utf-8')
    (output / 'questions.json').write_text(json.dumps({
        'title': title, 'episode': episode_number, 'episode_count': episode_count,
        'theme': theme, 'aspect': aspect, 'quality': quality, 'questions': saved,
    }, indent=2, ensure_ascii=False), encoding='utf-8')
    public.append('')
    public.append('Answers are on screen at the end of each question.')
    (output / 'description.txt').write_text('\n'.join(public) + '\n', encoding='utf-8')


def _episode_dir(root: Path, number: int) -> Path:
    return root / f'episode_{number:03d}'


def render_series(zip_paths, output: str | Path, *, per_type: int = 5, episodes: list[int] | None = None,
                  workers: int = 1, **settings) -> dict:
    """Build every full episode from the ZIPs and render the requested ones.

    `episodes` is a list of 1-based episode numbers. None renders all of them.
    """
    paths = resolve_zips(list(zip_paths) if zip_paths else None)
    groups = load_bank(paths)
    built = build_episodes(groups, per_type)
    if not built:
        smallest = min(groups, key=lambda name: len(groups[name]))
        raise ValueError(f'{category_label(smallest)} only has {len(groups[smallest])} questions. Need at least {per_type} of every type.')
    wanted = list(range(1, len(built) + 1)) if not episodes else [int(number) for number in episodes]
    for number in wanted:
        if number < 1 or number > len(built):
            raise ValueError(f'Episode {number} is outside 1–{len(built)}')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        'zips': [str(path) for path in paths],
        'per_type': per_type,
        'episode_count': len(built),
        'categories': present_categories(groups),
        'episodes': [{'episode': index, 'ids': [question['id'] for question in episode]} for index, episode in enumerate(built, 1)],
    }
    (output / 'series.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (output / 'series.txt').write_text(summary(groups, built, per_type) + '\n', encoding='utf-8')
    jobs = []
    for number in wanted:
        jobs.append({
            'episode': built[number - 1],
            'output': _episode_dir(output, number),
            'episode_number': number,
            'episode_count': len(built),
            **settings,
        })
    results = []
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_render_job, job): job['episode_number'] for job in jobs}
            for future in as_completed(futures):
                item = future.result()
                results.append(item)
                print(f"episode {item['episode']:03d} ready  {item['duration']}s  {item['video']}", flush=True)
    else:
        callback = settings.get('callback')
        for job in jobs:
            if callback:
                callback(0, f"Episode {job['episode_number']} of {len(built)}")
            item = render_episode(**job)
            results.append(item)
            print(f"episode {item['episode']:03d} ready  {item['duration']}s  {item['video']}", flush=True)
    results.sort(key=lambda item: item['episode'])
    return {'output': str(output), 'episodes': results, 'episode_count': len(built), 'rendered': len(results), 'summary': summary(groups, built, per_type)}


def _render_job(job: dict) -> dict:
    job = dict(job)
    job.pop('callback', None)
    return render_episode(**job)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Render visual trivia videos from ZIP question banks')
    parser.add_argument('--zip', action='append', dest='zips', help='Trivia ZIP. Repeat to use more than one. Default: every trivia ZIP in this folder.')
    parser.add_argument('--output', default='outputs/visual_trivia')
    parser.add_argument('--per-type', type=int, default=5, help='Questions from each puzzle type in every video')
    parser.add_argument('--episode', type=int, action='append', dest='episodes', help='1-based episode to render. Repeatable. Default with --all is every episode.')
    parser.add_argument('--all', action='store_true', help='Render every episode the ZIPs can fill')
    parser.add_argument('--quality', choices=['Preview', 'Full HD'], default='Preview')
    parser.add_argument('--aspect', choices=['9:16', '16:9'], default='9:16')
    parser.add_argument('--theme', default='Electric Midnight', choices=list(THEMES))
    parser.add_argument('--think', type=float, default=7.0, help='Seconds to think on a medium question. Easy is shorter, difficult is longer.')
    parser.add_argument('--answer', type=float, default=4.0, help='Seconds the answer stays on screen at the end of each question')
    parser.add_argument('--intro', type=float, default=2.0)
    parser.add_argument('--outro', type=float, default=2.2)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--no-sfx', action='store_true')
    args = parser.parse_args(argv)
    selected = None if args.all or not args.episodes else args.episodes
    if not args.all and not args.episodes:
        selected = [1]
    result = render_series(
        args.zips, args.output, per_type=args.per_type, episodes=selected, workers=max(1, args.workers),
        theme=args.theme, aspect=args.aspect, quality=args.quality, think=args.think, answer=args.answer,
        intro=args.intro, outro=args.outro, include_sfx=not args.no_sfx,
    )
    print(result['summary'])
    print(f"Rendered {result['rendered']} of {result['episode_count']} episodes into {result['output']}")
    for item in result['episodes']:
        print(f"  episode {item['episode']:03d}  {item['duration']}s  {item['resolution']}  {item['video']}")
    return result


if __name__ == '__main__':
    main()
