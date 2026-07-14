import struct
import sys

def main():
    r1cs_file = "dot_product.r1cs"
    print(f"[*] Reading {r1cs_file}...")
    try:
        with open(r1cs_file, "rb") as f:
            data = f.read()
    except FileNotFoundError:
        print(f"[!] Error: {r1cs_file} not found. Please compile the circuit first.")
        sys.exit(1)
    
    # 1. Magic
    magic = data[0:4]
    print(f"Magic: {magic.decode('ascii')} ({magic})")
    assert magic == b"r1cs", "Invalid magic"
    
    # 2. Version
    version = struct.unpack("<I", data[4:8])[0]
    print(f"Version: {version}")
    assert version == 1, "Invalid version"
    
    # 3. nSections
    n_sections = struct.unpack("<I", data[8:12])[0]
    print(f"Number of sections: {n_sections}")
    assert n_sections == 3, "Invalid nSections"
    
    offset = 12
    sections = {}
    for i in range(n_sections):
        sec_type = struct.unpack("<I", data[offset:offset+4])[0]
        sec_size = struct.unpack("<Q", data[offset+4:offset+12])[0]
        print(f"Section {i+1}: Type = {sec_type}, Size = {sec_size} bytes")
        sec_content = data[offset+12:offset+12+sec_size]
        sections[sec_type] = sec_content
        offset += 12 + sec_size
        
    # Header parse
    print("\n--- Header Section (Type 1) ---")
    header = sections[1]
    fs = struct.unpack("<I", header[0:4])[0]
    print(f"Field size (fs): {fs} bytes")
    assert fs == 32, "Field size must be 32"
    
    prime = int.from_bytes(header[4:4+32], byteorder="little")
    print(f"Prime Modulus: {prime}")
    assert prime == 21888242871839275222246405745257275088548364400416034343698204186575808495617, "Modulus mismatch"
    
    n_wires = struct.unpack("<I", header[36:40])[0]
    n_pub_out = struct.unpack("<I", header[40:44])[0]
    n_pub_in = struct.unpack("<I", header[44:48])[0]
    n_prv_in = struct.unpack("<I", header[48:52])[0]
    n_labels = struct.unpack("<Q", header[52:60])[0]
    m_constraints = struct.unpack("<I", header[60:64])[0]
    
    print(f"nWires: {n_wires}")
    print(f"nPubOut: {n_pub_out}")
    print(f"nPubIn: {n_pub_in}")
    print(f"nPrvIn: {n_prv_in}")
    print(f"nLabels: {n_labels}")
    print(f"mConstraints: {m_constraints}")
    
    assert n_wires == 100004, f"Expected 100004 wires, got {n_wires}"
    assert m_constraints == 100001, f"Expected 100001 constraints, got {m_constraints}"
    
    # Constraints parse
    print("\n--- Validating Constraints Section (Type 2) ---")
    constraints_data = sections[2]
    c_offset = 0
    
    # Let's count constraint sizes
    for idx in range(m_constraints):
        for lc_name in ["A", "B", "C"]:
            n_terms = struct.unpack("<I", constraints_data[c_offset:c_offset+4])[0]
            c_offset += 4
            for _ in range(n_terms):
                wire_id = struct.unpack("<I", constraints_data[c_offset:c_offset+4])[0]
                val = int.from_bytes(constraints_data[c_offset+4:c_offset+4+32], byteorder="little")
                c_offset += 36
    
    assert c_offset == len(constraints_data), f"Read {c_offset} bytes, but constraint section has {len(constraints_data)} bytes"
    print(f"Successfully traversed and verified all {m_constraints} constraints!")
    
    # Wire to Label Map parse
    print("\n--- Validating Wire2LabelId Map Section (Type 3) ---")
    map_data = sections[3]
    expected_map_size = n_wires * 8
    print(f"Wire map size: {len(map_data)} bytes (expected: {expected_map_size} bytes)")
    assert len(map_data) == expected_map_size, "Wire map size mismatch"
    
    print("\n[+] Verification successful! Binary is 100% correct and structurally valid.")

if __name__ == "__main__":
    main()
