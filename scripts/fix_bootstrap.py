import re

def main():
    file_path = "yc_bootstrap.c"
    print(f"[*] Reading {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Pattern to match GCC statement expression returning address of temporary:
    # ({ YStr __tmp = <expr>; &__tmp; })
    # We replace it with:
    # &((YStrUnion){.val = <expr>}).val
    pattern = r"\(\{ YStr __tmp = (.*?); &__tmp; \}\)"
    
    matches = re.findall(pattern, content)
    print(f"[*] Found {len(matches)} occurrences of statement expression temporaries.")
    
    if len(matches) > 0:
        new_content = re.sub(pattern, r"&((YStrUnion){.val = \1}).val", content)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"[+] Successfully patched {file_path}!")
    else:
        print("[*] No occurrences found or already patched.")

if __name__ == "__main__":
    main()
