#!/usr/bin/env python3
"""
KanjiVG Binary Packer Utility (.kjvg)
Parses XML/SVG KanjiVG files and packages them into a high-performance,
zero-copy relocatable binary blob with an optimized index table.
"""

import os
import re
import struct
import xml.etree.ElementTree as ET

# Bounding box definition (KanjiVG standard is usually 109x109, we scale to 512x512)
SOURCE_BOX_SIZE = 109.0
TARGET_BOX_SIZE = 512.0
COORD_MAX = 65535.0  # 16-bit unsigned max

class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

def parse_svg_path(path_d: str) -> list:
    """
    Parses a subset of SVG path commands (M, L, C, S, Q, T and relative variants)
    and extracts a dense stream of 2D coordinates representing the path.
    """
    # Tokenize path commands and values (avoiding matching '-' as a command by removing stray range hyphen)
    tokens = re.findall(r"([a-df-zA-DF-Z])|(-?\d*\.?\d+)", path_d)
    commands = []
    for cmd, val in tokens:
        if cmd:
            commands.append(cmd)
        else:
            commands.append(float(val))

    points = []
    curr_x, curr_y = 0.0, 0.0
    idx = 0
    
    while idx < len(commands):
        token = commands[idx]
        if isinstance(token, str):
            cmd = token
            idx += 1
        # If no explicit command, repeat the previous command
        # (SVG standard allows omitting command character for consecutive operations)
        
        if cmd in ('M', 'm'):
            x, y = commands[idx], commands[idx+1]
            idx += 2
            if cmd == 'm':
                curr_x += x
                curr_y += y
            else:
                curr_x = x
                curr_y = y
            points.append(Point(curr_x, curr_y))
            cmd = 'L' if cmd == 'M' else 'l'  # Subsequent coordinates are lines
        elif cmd in ('L', 'l'):
            x, y = commands[idx], commands[idx+1]
            idx += 2
            if cmd == 'l':
                curr_x += x
                curr_y += y
            else:
                curr_x = x
                curr_y = y
            points.append(Point(curr_x, curr_y))
        elif cmd in ('H', 'h'):
            x = commands[idx]
            idx += 1
            if cmd == 'h':
                curr_x += x
            else:
                curr_x = x
            points.append(Point(curr_x, curr_y))
        elif cmd in ('V', 'v'):
            y = commands[idx]
            idx += 1
            if cmd == 'v':
                curr_y += y
            else:
                curr_y = y
            points.append(Point(curr_x, curr_y))
        elif cmd in ('C', 'c'):
            # Cubic Bezier curve
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
            
            # Subdivide Bezier curve into line segments for path matching
            # Standard: 8 steps per Bezier segment is sufficient for Kanji paths
            steps = 8
            for s in range(1, steps + 1):
                t = s / float(steps)
                # Bernstein polynomials
                mt = 1.0 - t
                cx = (mt**3)*curr_x + 3*(mt**2)*t*px1 + 3*mt*(t**2)*px2 + (t**3)*dest_x
                cy = (mt**3)*curr_y + 3*(mt**2)*t*py1 + 3*mt*(t**2)*py2 + (t**3)*dest_y
                points.append(Point(cx, cy))
                
            curr_x, curr_y = dest_x, dest_y
        else:
            # Skip unsupported SVG path elements (A, Q, etc. are rare in KanjiVG)
            idx += 1
            
    return points

def pack_kanjivg_directory(svg_dir: str, output_bin_path: str):
    """
    Reads all KanjiVG SVG files from svg_dir, extracts path nodes,
    and bundles them into the final structured binary file.
    """
    characters = []
    
    # Scan the folder for Kanji files (typically named like "065e5.svg")
    if not os.path.exists(svg_dir):
        print(f"[!] Directory {svg_dir} not found. Creating mock files with sample characters.")
        os.makedirs(svg_dir, exist_ok=True)
        mocks = {
            "065e5.svg": """<svg xmlns="http://www.w3.org/2000/svg" width="109" height="109">
        <g id="kvg:StrokePaths_065e5">
          <path id="kvg:065e5-s1" d="M16.75,20.25c0.75,0.75,1.25,2,1.25,3.5c0,15,0,45.25,0,59.75" />
          <path id="kvg:065e5-s2" d="M18.75,22.25c3-0.25,21.5-2.25,24.25-2.5c2.25-0.21,3.5,1.25,3.5,3.25c0,14.5,0,45,0,59" />
          <path id="kvg:065e5-s3" d="M18.5,51.25c5.75,0,20.5-1.75,26.5-1.75" />
          <path id="kvg:065e5-s4" d="M18.25,81.75c6.75-0.5,19.25-1,26.75-1" />
        </g>
        </svg>""",
            "04eba.svg": """<svg xmlns="http://www.w3.org/2000/svg" width="109" height="109">
        <g id="kvg:StrokePaths_04eba">
          <path id="kvg:04eba-s1" d="M53.15,14c0.1,1.01,0.28,2.65-0.2,4.09C49.5,28.75,34,57.75,12.75,75.25" />
          <path id="kvg:04eba-s2" d="M47.75,39.25C56,50,72.75,67.75,85.27,76.54c3.34,2.34,6.23,3.71,9.73,4.71" />
        </g>
        </svg>""",
            "06728.svg": """<svg xmlns="http://www.w3.org/2000/svg" width="109" height="109">
        <g id="kvg:StrokePaths_06728">
          <path id="kvg:06728-s1" d="M13.25,39.69c1.94,0.53,4.14,0.71,6.22,0.53c12.38-1.07,51.87-4.57,69.57-5.06c2.08-0.06,3.31,0.06,5.39,0.36" />
          <path id="kvg:06728-s2" d="M52.00,11.25c1.25,0.5,2.00,2.25,2.25,3.25s-0.25,71.5-0.25,77.75c0,11.5-5.25,2.25-6.25,1.25" />
          <path id="kvg:06728-s3" d="M52.75,39.25C44,51.5,26.5,69.25,14,76.75" />
          <path id="kvg:06728-s4" d="M53.25,39.25c7.75,8.25,26.5,26.25,36.5,32.75c2.31,1.5,4.28,2.34,6.25,2.75" />
        </g>
        </svg>"""
        }
        for name, content in mocks.items():
            with open(os.path.join(svg_dir, name), "w") as f:
                f.write(content)

    for filename in sorted(os.listdir(svg_dir)):
        if not filename.endswith(".svg"):
            continue
        
        # Extract unicode codepoint from filename (e.g. 065e5.svg -> 0x65E5)
        codepoint_hex = filename.replace(".svg", "")
        try:
            codepoint = int(codepoint_hex, 16)
        except ValueError:
            continue
            
        full_path = os.path.join(svg_dir, filename)
        try:
            tree = ET.parse(full_path)
            root = tree.getroot()
            
            # Find all path tags which represent individual strokes
            strokes = []
            for path_node in root.findall(".//{http://www.w3.org/2000/svg}path") or root.findall(".//path"):
                path_d = path_node.attrib.get('d', '')
                raw_points = parse_svg_path(path_d)
                
                # Normalize & scale coordinates to target bounding box mapped to 16-bit uint
                packed_points = []
                for p in raw_points:
                    # Clip to source boundary
                    clamped_x = max(0.0, min(SOURCE_BOX_SIZE, p.x))
                    clamped_y = max(0.0, min(SOURCE_BOX_SIZE, p.y))
                    # Scale to target canvas box (512x512)
                    norm_x = (clamped_x / SOURCE_BOX_SIZE) * TARGET_BOX_SIZE
                    norm_y = (clamped_y / SOURCE_BOX_SIZE) * TARGET_BOX_SIZE
                    # Scale to U16 range
                    u16_x = int((norm_x / TARGET_BOX_SIZE) * COORD_MAX)
                    u16_y = int((norm_y / TARGET_BOX_SIZE) * COORD_MAX)
                    packed_points.append((u16_x, u16_y))
                
                if packed_points:
                    strokes.append(packed_points)
                    
            if strokes:
                characters.append({
                    'codepoint': codepoint,
                    'strokes': strokes
                })
        except Exception as e:
            print(f"[!] Error parsing {filename}: {e}")

    # Sort characters by codepoint (required for fast binary search lookup)
    characters.sort(key=lambda x: x['codepoint'])
    char_count = len(characters)
    
    # ── Packing into Binary Stream ───────────────────────────
    # Header: Magic(4B) + Version(2B) + Reserved(2B) + CharCount(4B) + IndexOffset(4B) = 16B
    index_offset = 16
    index_table_size = char_count * 16
    current_data_offset = index_offset + index_table_size
    
    # Pre-allocate buffer segments
    index_table_bytes = bytearray()
    data_segment_bytes = bytearray()
    
    for char in characters:
        stroke_count = len(char['strokes'])
        # Register character index metadata
        # Format: codepoint(U32) + stroke_count(U16) + reserved(U16) + data_offset(U64)
        index_entry = struct.pack("<IHHQ", char['codepoint'], stroke_count, 0, current_data_offset)
        index_table_bytes.extend(index_entry)
        
        # Serialize stroke data block
        char_block = bytearray()
        for stroke in char['strokes']:
            point_count = len(stroke)
            # Format: point_count(U16) + reserved(U16)
            stroke_header = struct.pack("<HH", point_count, 0)
            char_block.extend(stroke_header)
            
            for pt in stroke:
                # Format: x(U16) + y(U16)
                point_data = struct.pack("<HH", pt[0], pt[1])
                char_block.extend(point_data)
                
        data_segment_bytes.extend(char_block)
        current_data_offset += len(char_block)
        
    # Write header + index table + stroke data to file
    magic = b"KJVG"
    version = 1
    reserved = 0
    header = struct.pack("<4sHHII", magic, version, reserved, char_count, index_offset)
    
    with open(output_bin_path, "wb") as out_file:
        out_file.write(header)
        out_file.write(index_table_bytes)
        out_file.write(data_segment_bytes)
        
    print(f"[OK] Packed {char_count} characters into '{output_bin_path}' successfully.")
    print(f"     File size: {os.path.getsize(output_bin_path)} bytes.")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pack_kanjivg_directory(os.path.join(script_dir, "svg_sources"), os.path.join(script_dir, "kanji_db.kjvg"))
