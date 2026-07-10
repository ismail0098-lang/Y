#!/usr/bin/env python3
import os
import re
import gzip
import json
import struct
import urllib.request
import xml.etree.ElementTree as ET

KANJIVG_URL = "https://github.com/KanjiVG/kanjivg/releases/download/r20220427/kanjivg-20220427.xml.gz"
KANJIDIC_URL = "https://www.edrdg.org/kanjidic/kanjidic2.xml.gz"

SOURCE_BOX_SIZE = 109.0
TARGET_BOX_SIZE = 512.0
COORD_MAX = 65535.0

class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

def parse_svg_path(path_d: str) -> list:
    tokens = re.findall(r"([a-df-zA-DF-Z])|(-?\d*\.?\d+)", path_d)
    commands = []
    for cmd, val in tokens:
        if cmd:
            commands.append(cmd)
        else:
            commands.append(float(val))

    points = []
    curr_x, curr_y = 0.0, 0.0
    prev_x2, prev_y2 = None, None  # For S/s cubic Bezier reflection
    prev_x1, prev_y1 = None, None  # For T/t quadratic Bezier reflection
    last_cmd = ''
    idx = 0
    cmd = ''
    
    while idx < len(commands):
        token = commands[idx]
        if isinstance(token, str):
            cmd = token
            idx += 1
        
        if cmd in ('M', 'm'):
            if idx + 1 >= len(commands): break
            x, y = commands[idx], commands[idx+1]
            idx += 2
            if cmd == 'm':
                curr_x += x
                curr_y += y
            else:
                curr_x = x
                curr_y = y
            points.append(Point(curr_x, curr_y))
            prev_x2, prev_y2 = None, None
            prev_x1, prev_y1 = None, None
            last_cmd = cmd
            cmd = 'L' if cmd == 'M' else 'l'
        elif cmd in ('L', 'l'):
            if idx + 1 >= len(commands): break
            x, y = commands[idx], commands[idx+1]
            idx += 2
            if cmd == 'l':
                curr_x += x
                curr_y += y
            else:
                curr_x = x
                curr_y = y
            points.append(Point(curr_x, curr_y))
            prev_x2, prev_y2 = None, None
            prev_x1, prev_y1 = None, None
            last_cmd = cmd
        elif cmd in ('H', 'h'):
            if idx >= len(commands): break
            x = commands[idx]
            idx += 1
            if cmd == 'h':
                curr_x += x
            else:
                curr_x = x
            points.append(Point(curr_x, curr_y))
            prev_x2, prev_y2 = None, None
            prev_x1, prev_y1 = None, None
            last_cmd = cmd
        elif cmd in ('V', 'v'):
            if idx >= len(commands): break
            y = commands[idx]
            idx += 1
            if cmd == 'v':
                curr_y += y
            else:
                curr_y = y
            points.append(Point(curr_x, curr_y))
            prev_x2, prev_y2 = None, None
            prev_x1, prev_y1 = None, None
            last_cmd = cmd
        elif cmd in ('C', 'c'):
            if idx + 5 >= len(commands): break
            x1, y1 = commands[idx], commands[idx+1]
            x2, y2 = commands[idx+2], commands[idx+3]
            x, y = commands[idx+4], commands[idx+5]
            idx += 6
            
            if cmd == 'c':
                px1, py1 = curr_x + x1, curr_y + y1
                px2, py2 = curr_x + x2, curr_y + y2
                dest_x, dest_y = curr_x + x, curr_y + y
            else:
                px1, py1 = x1, y1
                px2, py2 = x2, y2
                dest_x, dest_y = x, y
            
            steps = 8
            for s in range(1, steps + 1):
                t = s / float(steps)
                mt = 1.0 - t
                cx = (mt**3)*curr_x + 3*(mt**2)*t*px1 + 3*mt*(t**2)*px2 + (t**3)*dest_x
                cy = (mt**3)*curr_y + 3*(mt**2)*t*py1 + 3*mt*(t**2)*py2 + (t**3)*dest_y
                points.append(Point(cx, cy))
                
            curr_x, curr_y = dest_x, dest_y
            prev_x2, prev_y2 = px2, py2
            prev_x1, prev_y1 = None, None
            last_cmd = cmd
        elif cmd in ('S', 's'):
            if idx + 3 >= len(commands): break
            x2, y2 = commands[idx], commands[idx+1]
            x, y = commands[idx+2], commands[idx+3]
            idx += 4
            
            if cmd == 's':
                px2, py2 = curr_x + x2, curr_y + y2
                dest_x, dest_y = curr_x + x, curr_y + y
            else:
                px2, py2 = x2, y2
                dest_x, dest_y = x, y
            
            if last_cmd in ('C', 'c', 'S', 's') and prev_x2 is not None and prev_y2 is not None:
                px1 = 2 * curr_x - prev_x2
                py1 = 2 * curr_y - prev_y2
            else:
                px1 = curr_x
                py1 = curr_y
                
            steps = 8
            for s in range(1, steps + 1):
                t = s / float(steps)
                mt = 1.0 - t
                cx = (mt**3)*curr_x + 3*(mt**2)*t*px1 + 3*mt*(t**2)*px2 + (t**3)*dest_x
                cy = (mt**3)*curr_y + 3*(mt**2)*t*py1 + 3*mt*(t**2)*py2 + (t**3)*dest_y
                points.append(Point(cx, cy))
                
            curr_x, curr_y = dest_x, dest_y
            prev_x2, prev_y2 = px2, py2
            prev_x1, prev_y1 = None, None
            last_cmd = cmd
        elif cmd in ('Q', 'q'):
            if idx + 3 >= len(commands): break
            x1, y1 = commands[idx], commands[idx+1]
            x, y = commands[idx+2], commands[idx+3]
            idx += 4
            
            if cmd == 'q':
                px1, py1 = curr_x + x1, curr_y + y1
                dest_x, dest_y = curr_x + x, curr_y + y
            else:
                px1, py1 = x1, y1
                dest_x, dest_y = x, y
                
            steps = 8
            for s in range(1, steps + 1):
                t = s / float(steps)
                mt = 1.0 - t
                cx = (mt**2)*curr_x + 2*mt*t*px1 + (t**2)*dest_x
                cy = (mt**2)*curr_y + 2*mt*t*py1 + (t**2)*dest_y
                points.append(Point(cx, cy))
                
            curr_x, curr_y = dest_x, dest_y
            prev_x2, prev_y2 = None, None
            prev_x1, prev_y1 = px1, py1
            last_cmd = cmd
        elif cmd in ('T', 't'):
            if idx + 1 >= len(commands): break
            x, y = commands[idx], commands[idx+1]
            idx += 2
            
            if cmd == 't':
                dest_x, dest_y = curr_x + x, curr_y + y
            else:
                dest_x, dest_y = x, y
                
            if last_cmd in ('Q', 'q', 'T', 't') and prev_x1 is not None and prev_y1 is not None:
                px1 = 2 * curr_x - prev_x1
                py1 = 2 * curr_y - prev_y1
            else:
                px1 = curr_x
                py1 = curr_y
                
            steps = 8
            for s in range(1, steps + 1):
                t = s / float(steps)
                mt = 1.0 - t
                cx = (mt**2)*curr_x + 2*mt*t*px1 + (t**2)*dest_x
                cy = (mt**2)*curr_y + 2*mt*t*py1 + (t**2)*dest_y
                points.append(Point(cx, cy))
                
            curr_x, curr_y = dest_x, dest_y
            prev_x2, prev_y2 = None, None
            prev_x1, prev_y1 = px1, py1
            last_cmd = cmd
        else:
            idx += 1
            
    return points

def download_file(url, filename):
    if os.path.exists(filename):
        print(f"[+] '{filename}' already exists, skipping download.")
        return
    print(f"[-] Downloading {url}...")
    urllib.request.urlretrieve(url, filename)
    print(f"[+] Download complete: {filename}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    kanjivg_gz = os.path.join(script_dir, "kanjivg.xml.gz")
    kanjidic_gz = os.path.join(script_dir, "kanjidic2.xml.gz")

    # 1. Download source files
    download_file(KANJIVG_URL, kanjivg_gz)
    download_file(KANJIDIC_URL, kanjidic_gz)

    # 2. Parse KANJIDIC2
    print("[-] Parsing KANJIDIC2...")
    kanji_meta = {}
    with gzip.open(kanjidic_gz, 'rb') as f:
        # XML Parser
        context = ET.iterparse(f, events=('end',))
        for event, elem in context:
            if elem.tag == 'character':
                literal_node = elem.find('literal')
                if literal_node is not None:
                    char = literal_node.text
                    meanings = []
                    onyomi = []
                    kunyomi = []
                    
                    # Extract readings & meanings
                    rmgroup = elem.find('.//rmgroup')
                    if rmgroup is not None:
                        for m in rmgroup.findall('meaning'):
                            # Only English meanings (no lang attribute)
                            if 'm_lang' not in m.attrib:
                                meanings.append(m.text)
                        for r in rmgroup.findall('reading'):
                            r_type = r.attrib.get('r_type')
                            if r_type == 'ja_on':
                                onyomi.append(r.text)
                            elif r_type == 'ja_kun':
                                kunyomi.append(r.text)
                    
                    kanji_meta[char] = {
                        "literal": char,
                        "meanings": meanings[:6],  # Keep up to 6 meanings to optimize size
                        "onyomi": onyomi,
                        "kunyomi": kunyomi
                    }
                # Clean elements from memory to keep footprint minimal
                elem.clear()

    print(f"[+] Loaded dictionary details for {len(kanji_meta)} characters.")

    # 3. Parse KanjiVG XML
    print("[-] Parsing KanjiVG paths...")
    characters = []
    
    with gzip.open(kanjivg_gz, 'rb') as f:
        tree = ET.parse(f)
        root = tree.getroot()
        
        # Iterate all <kanji> tags
        for kanji_node in root.findall('.//kanji'):
            kvg_id = kanji_node.attrib.get('id')
            if not kvg_id: continue
            
            # Format: 'kvg:kanji_065e5' -> extract codepoint hex
            codepoint_hex = kvg_id.split('_')[-1]
            try:
                codepoint = int(codepoint_hex, 16)
            except ValueError:
                continue
                
            char = chr(codepoint)
            
            # Find all stroke path nodes
            strokes = []
            for path_node in kanji_node.findall('.//path'):
                path_d = path_node.attrib.get('d', '')
                raw_points = parse_svg_path(path_d)
                
                packed_points = []
                for p in raw_points:
                    clamped_x = max(0.0, min(SOURCE_BOX_SIZE, p.x))
                    clamped_y = max(0.0, min(SOURCE_BOX_SIZE, p.y))
                    norm_x = (clamped_x / SOURCE_BOX_SIZE) * TARGET_BOX_SIZE
                    norm_y = (clamped_y / SOURCE_BOX_SIZE) * TARGET_BOX_SIZE
                    u16_x = int((norm_x / TARGET_BOX_SIZE) * COORD_MAX)
                    u16_y = int((norm_y / TARGET_BOX_SIZE) * COORD_MAX)
                    packed_points.append((u16_x, u16_y))
                
                if packed_points:
                    strokes.append(packed_points)
            
            if strokes:
                characters.append({
                    'codepoint': codepoint,
                    'literal': char,
                    'strokes': strokes
                })

    print(f"[+] Loaded drawings for {len(characters)} characters.")

    # 4. Filter metadata to only include packed KanjiVG characters to save space
    packed_codepoints = {c['codepoint'] for c in characters}
    filtered_meta = {}
    for codepoint in packed_codepoints:
        char = chr(codepoint)
        meta = kanji_meta.get(char)
        if meta:
            filtered_meta[f"0x{codepoint:04X}"] = meta
        else:
            # Fallback if dictionary has no entry
            filtered_meta[f"0x{codepoint:04X}"] = {
                "literal": char,
                "meanings": ["No dictionary definition"],
                "onyomi": [],
                "kunyomi": []
            }

    # 5. Pack into Binary Stream (.kjvg)
    characters.sort(key=lambda x: x['codepoint'])
    char_count = len(characters)
    
    index_offset = 16
    index_table_size = char_count * 16
    current_data_offset = index_offset + index_table_size
    
    index_table_bytes = bytearray()
    data_segment_bytes = bytearray()
    
    for char in characters:
        codepoint = char['codepoint']
        strokes = char['strokes']
        stroke_count = len(strokes)
        
        index_table_bytes.extend(struct.pack("<IHHQ", codepoint, stroke_count, 0, current_data_offset))
        
        char_data = bytearray()
        for stroke in strokes:
            pt_count = len(stroke)
            char_data.extend(struct.pack("<HH", pt_count, 0))
            for x, y in stroke:
                char_data.extend(struct.pack("<HH", x, y))
        
        data_segment_bytes.extend(char_data)
        current_data_offset += len(char_data)

    output_bin = os.path.join(script_dir, "kanji_db.kjvg")
    output_json = os.path.join(script_dir, "kanji_meta.json")

    with open(output_bin, "wb") as f:
        f.write(struct.pack("<4sHHI I", b"KJVG", 1, 0, char_count, index_offset))
        f.write(index_table_bytes)
        f.write(data_segment_bytes)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(filtered_meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] Successfully built files:")
    print(f"     Binary: {output_bin} ({os.path.getsize(output_bin)} bytes, {char_count} characters)")
    print(f"     Metadata: {output_json} ({os.path.getsize(output_json)} bytes)")

    # 6. Copy files to frontend folder
    frontend_dir = os.path.join(os.path.dirname(script_dir), "frontend")
    if os.path.exists(frontend_dir):
        dest_bin = os.path.join(frontend_dir, "kanji_db.kjvg")
        dest_json = os.path.join(frontend_dir, "kanji_meta.json")
        with open(output_bin, "rb") as f_src, open(dest_bin, "wb") as f_dest:
            f_dest.write(f_src.read())
        with open(output_json, "r", encoding="utf-8") as f_src, open(dest_json, "w", encoding="utf-8") as f_dest:
            f_dest.write(f_src.read())
        print(f"[OK] Copied files to frontend workspace directory successfully.")

if __name__ == "__main__":
    main()
